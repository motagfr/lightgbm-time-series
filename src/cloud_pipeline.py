"""
Multi-Cloud Data Pipeline & Model Registry (Google Cloud Platform & Azure)
-------------------------------------------------------------------------
Provides automated artifact persistence, dataset export, and model deployment
to both Google Cloud Platform (GCS / BigQuery) and Microsoft Azure (Blob Storage / Azure ML).
"""

import os
import json
import logging
import warnings
import argparse
import pandas as pd
from typing import Dict, Any, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Mute SDK logging noise during local runs
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.identity").setLevel(logging.ERROR)
logging.getLogger("azure.core").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# GCP Imports
try:
    from google.cloud import storage, bigquery
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Azure Imports
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class GCPPipelineManager:
    """Manages model artifacts and time series predictions on Google Cloud Platform."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        if not GCP_AVAILABLE:
            raise ImportError("google-cloud-storage and google-cloud-bigquery must be installed.")
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)

    def upload_model_artifact(self, local_path: str, bucket_name: str, destination_blob: str):
        """Upload trained LightGBM model, predictions, or parquet metrics to Google Cloud Storage."""
        bucket = self.storage_client.bucket(bucket_name)
        if not bucket.exists():
            print(f"[GCP Pipeline] Local Mode: Remote bucket gs://{bucket_name} is not provisioned. Artifact saved locally to {local_path}.")
            return
        print(f"[GCP] Uploading {local_path} to gs://{bucket_name}/{destination_blob}...")
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)
        print(f"[GCP] Successfully uploaded artifact to gs://{bucket_name}/{destination_blob}")

    def export_predictions_to_bigquery(self, df: pd.DataFrame, dataset_id: str, table_id: str):
        """Publish time series predictions into a BigQuery table."""
        full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
        print(f"[GCP] Streaming predictions to BigQuery table: {full_table_id}...")
        
        # Format date column for BigQuery schema
        bq_df = df.copy()
        if "date" in bq_df.columns:
            bq_df["date"] = pd.to_datetime(bq_df["date"]).dt.date

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.bq_client.load_table_from_dataframe(bq_df, full_table_id, job_config=job_config)
        job.result()
        print(f"[GCP] Loaded {len(bq_df)} rows into BigQuery table {full_table_id}")


class AzurePipelineManager:
    """Manages model artifacts and time series predictions on Microsoft Azure."""

    def __init__(self, connection_string: Optional[str] = None):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob and azure-identity must be installed.")
        
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        else:
            if not any(k in os.environ for k in ["AZURE_CLIENT_ID", "AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_KEY", "AZURE_TENANT_ID"]):
                raise ValueError("No Azure storage credentials or connection string found in environment variables.")
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "https://stlightgbmmotagfr.blob.core.windows.net")
            credential = DefaultAzureCredential(logging_enable=False)
            self.blob_service_client = BlobServiceClient(account_url, credential=credential)

    def upload_model_artifact(self, local_path: str, container_name: str, destination_blob: str):
        """Upload trained LightGBM model or parquet metrics to Azure Blob Storage."""
        container_client = self.blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            print(f"[Azure Pipeline] Local Mode: Remote container '{container_name}' is not provisioned. Artifact saved locally to {local_path}.")
            return
        print(f"[Azure] Uploading {local_path} to container '{container_name}' as '{destination_blob}'...")
        blob_client = container_client.get_blob_client(destination_blob)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"[Azure] Successfully uploaded artifact to Azure Blob: {container_name}/{destination_blob}")


def export_tuned_model_and_predictions(gcp_project: str = "booming-edge-452110-g8", gcp_bucket: str = "lightgbm-models-motagfr", azure_container: str = "lightgbm-models"):
    """Export final tuned LightGBM model weights and prediction dataframes to GCP and Azure."""
    from src.optuna_tune import run_tuning
    
    print("\n--- Running Optuna Tuning & Generating Final Artifacts ---")
    study, tuned_model = run_tuning(n_trials=20)
    
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Save tuned model weights binary
    model_path = "artifacts/lightgbm_model_tuned.txt"
    tuned_model.booster_.save_model(model_path)
    print(f"Saved tuned model weights to {model_path}")
    
    # 2. Generate and save prediction dataframe
    from src.forecast_demo import generate_synthetic_time_series, create_time_series_features
    raw_df = generate_synthetic_time_series(n_days=730)
    featured_df = create_time_series_features(raw_df)
    feature_cols = [c for c in featured_df.columns if c not in ["date", "sales"]]
    
    split_idx = int(len(featured_df) * 0.8)
    test_df = featured_df.iloc[split_idx:].copy()
    test_df["predicted_sales"] = tuned_model.predict(test_df[feature_cols])
    test_df["error"] = test_df["sales"] - test_df["predicted_sales"]
    
    pred_path = "artifacts/forecast_predictions.parquet"
    test_df[["date", "sales", "predicted_sales", "error"]].to_parquet(pred_path, index=False)
    print(f"Saved predictions dataframe to {pred_path}")
    
    # 3. Save tuned metrics json
    metrics_path = "artifacts/model_metrics_tuned.json"
    metrics_data = {
        "model": "Optuna-Tuned LightGBM Regressor",
        "best_rmse": round(float(study.best_value), 4),
        "best_hyperparameters": study.best_params,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved tuned metrics json to {metrics_path}")
    
    # 4. GCP Export (Storage + BigQuery)
    print("\n--- GCP Multi-Cloud Export ---")
    try:
        gcp_mgr = GCPPipelineManager(project_id=gcp_project)
        gcp_mgr.upload_model_artifact(model_path, gcp_bucket, "tuned/lightgbm_model_tuned.txt")
        gcp_mgr.upload_model_artifact(pred_path, gcp_bucket, "tuned/forecast_predictions.parquet")
        gcp_mgr.upload_model_artifact(metrics_path, gcp_bucket, "tuned/model_metrics_tuned.json")
        gcp_mgr.export_predictions_to_bigquery(test_df[["date", "sales", "predicted_sales", "error"]], "lightgbm_forecasts", "tuned_predictions")
    except Exception as e:
        print(f"[GCP Export Info] {e}")
        
    # 5. Azure Export
    print("\n--- Azure Multi-Cloud Export ---")
    try:
        azure_mgr = AzurePipelineManager(connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        azure_mgr.upload_model_artifact(model_path, azure_container, "tuned/lightgbm_model_tuned.txt")
        azure_mgr.upload_model_artifact(pred_path, azure_container, "tuned/forecast_predictions.parquet")
        azure_mgr.upload_model_artifact(metrics_path, azure_container, "tuned/model_metrics_tuned.json")
    except Exception as e:
        print(f"[Azure Export Info] Local Mode: Azure storage unconfigured.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Cloud Model Pipeline (GCP & Azure)")
    parser.add_argument("--provider", choices=["gcp", "azure", "both"], default="both", help="Cloud provider target")
    parser.add_argument("--gcp-project", default="booming-edge-452110-g8", help="GCP Project ID")
    parser.add_argument("--gcp-bucket", default="lightgbm-models-motagfr", help="GCP Storage Bucket")
    parser.add_argument("--azure-container", default="lightgbm-models", help="Azure Storage Container")
    
    args = parser.parse_args()
    export_tuned_model_and_predictions(gcp_project=args.gcp_project, gcp_bucket=args.gcp_bucket, azure_container=args.azure_container)


if __name__ == "__main__":
    main()
