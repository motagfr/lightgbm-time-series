"""
Multi-Cloud Data Pipeline & Model Registry (Google Cloud Platform & Azure)
-------------------------------------------------------------------------
Provides automated artifact persistence, dataset export, and model deployment
to both Google Cloud Platform (GCS / BigQuery) and Microsoft Azure (Blob Storage / Azure ML).
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, Any, Optional

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
        """Upload trained LightGBM model or parquet metrics to Google Cloud Storage."""
        print(f"[GCP] Uploading {local_path} to gs://{bucket_name}/{destination_blob}...")
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)
        print(f"[GCP] Successfully uploaded artifact to gs://{bucket_name}/{destination_blob}")

    def export_predictions_to_bigquery(self, df: pd.DataFrame, dataset_id: str, table_id: str):
        """Publish time series predictions into a BigQuery table."""
        full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
        print(f"[GCP] Streaming predictions to BigQuery table: {full_table_id}...")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        job.result()
        print(f"[GCP] Loaded {len(df)} rows into {full_table_id}")


class AzurePipelineManager:
    """Manages model artifacts and time series predictions on Microsoft Azure."""

    def __init__(self, connection_string: Optional[str] = None):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob and azure-identity must be installed.")
        
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use Azure Default Credentials (Environment, Managed Identity, Azure CLI)
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "https://stlightgbmmotagfr.blob.core.windows.net")
            credential = DefaultAzureCredential()
            self.blob_service_client = BlobServiceClient(account_url, credential=credential)

    def upload_model_artifact(self, local_path: str, container_name: str, destination_blob: str):
        """Upload trained LightGBM model or parquet metrics to Azure Blob Storage."""
        print(f"[Azure] Uploading {local_path} to container '{container_name}' as '{destination_blob}'...")
        container_client = self.blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            
        blob_client = container_client.get_blob_client(destination_blob)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"[Azure] Successfully uploaded artifact to Azure Blob: {container_name}/{destination_blob}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Cloud Model Pipeline (GCP & Azure)")
    parser.add_argument("--provider", choices=["gcp", "azure", "both"], default="both", help="Cloud provider target")
    parser.add_argument("--gcp-project", default="booming-edge-452110-g8", help="GCP Project ID")
    parser.add_argument("--gcp-bucket", default="lightgbm-models-motagfr", help="GCP Storage Bucket")
    parser.add_argument("--azure-container", default="lightgbm-models", help="Azure Storage Container")
    
    args = parser.parse_args()
    
    # Save a sample metrics artifact for demonstration
    sample_metrics = {
        "model": "LightGBM Regressor",
        "task": "Time Series Forecasting",
        "rmse": 8.4606,
        "mae": 6.5768,
        "mape_percent": 4.56,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    os.makedirs("artifacts", exist_ok=True)
    metrics_path = "artifacts/model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(sample_metrics, f, indent=2)
        
    print(f"Generated metrics artifact at {metrics_path}")
    print(json.dumps(sample_metrics, indent=2))
    
    if args.provider in ["gcp", "both"]:
        print("\n--- GCP Pipeline Execution ---")
        try:
            gcp_mgr = GCPPipelineManager(project_id=args.gcp_project)
            gcp_mgr.upload_model_artifact(metrics_path, args.gcp_bucket, "metrics/latest_metrics.json")
        except Exception as e:
            print(f"[GCP Pipeline Info] Cloud upload skipped (Local / Offline mode): {e}")
            print("[GCP Pipeline Info] To enable live GCP storage, ensure billing is active and project bucket exists.")

    if args.provider in ["azure", "both"]:
        print("\n--- Azure Pipeline Execution ---")
        try:
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            azure_mgr = AzurePipelineManager(connection_string=conn_str)
            azure_mgr.upload_model_artifact(metrics_path, args.azure_container, "metrics/latest_metrics.json")
        except Exception as e:
            print(f"[Azure Pipeline Info] Cloud upload skipped (Local / Offline mode): Azure credentials not configured.")
            print("[Azure Pipeline Info] Set AZURE_STORAGE_CONNECTION_STRING or run 'az login' to enable live Azure uploads.")



if __name__ == "__main__":
    main()
