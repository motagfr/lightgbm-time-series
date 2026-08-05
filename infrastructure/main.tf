# Terraform Multi-Cloud Infrastructure (Google Cloud Platform & Microsoft Azure)
# Author: @motagfr

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
}

# ------------------------------------------------------------------------------
# Provider Configurations
# ------------------------------------------------------------------------------
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "azurerm" {
  features {}
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
variable "gcp_project_id" {
  type        = string
  default     = "booming-edge-452110-g8"
  description = "Google Cloud Platform Project ID"
}

variable "gcp_region" {
  type        = string
  default     = "us-central1"
  description = "Default GCP Region"
}

variable "azure_location" {
  type        = string
  default     = "East US"
  description = "Default Azure Location"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Environment tag"
}

# ------------------------------------------------------------------------------
# Google Cloud Platform (GCP) Infrastructure
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "gcp_model_bucket" {
  name                     = "lightgbm-models-motagfr"
  location                 = var.gcp_region
  force_destroy            = true
  public_access_prevention = "enforced"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "lightgbm-time-series"
  }
}

resource "google_bigquery_dataset" "time_series_dataset" {
  dataset_id                  = "lightgbm_forecasts"
  friendly_name               = "LightGBM Time Series Forecasts"
  description                 = "Stores time-series predictions and historical evaluations."
  location                    = var.gcp_region
  default_table_expiration_ms = 3600000 * 24 * 365 # 1 Year

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ------------------------------------------------------------------------------
# Microsoft Azure Infrastructure
# ------------------------------------------------------------------------------
resource "azurerm_resource_group" "azure_rg" {
  name     = "rg-lightgbm-forecasting"
  location = var.azure_location

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "motagfr"
  }
}

resource "azurerm_storage_account" "azure_sa" {
  name                     = "stlightgbmmotagfr"
  resource_group_name      = azurerm_resource_group.azure_rg.name
  location                 = azurerm_resource_group.azure_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  min_tls_version = "TLS1_2"

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "azurerm_storage_container" "azure_model_container" {
  name                  = "lightgbm-models"
  storage_account_name  = azurerm_storage_account.azure_sa.name
  container_access_type = "private"
}

# Azure Machine Learning Workspace
resource "azurerm_application_insights" "azure_app_insights" {
  name                = "appi-lightgbm"
  location            = azurerm_resource_group.azure_rg.location
  resource_group_name = azurerm_resource_group.azure_rg.name
  application_type    = "web"
}

resource "azurerm_key_vault" "azure_kv" {
  name                        = "kv-lightgbm-motagfr"
  location                    = azurerm_resource_group.azure_rg.location
  resource_group_name         = azurerm_resource_group.azure_rg.name
  tenant_id                   = "00000000-0000-0000-0000-000000000000" # Placeholder for tenant ID
  sku_name                    = "standard"
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
}

resource "azurerm_machine_learning_workspace" "azure_ml_workspace" {
  name                    = "aml-lightgbm-workspace"
  location                = azurerm_resource_group.azure_rg.location
  resource_group_name     = azurerm_resource_group.azure_rg.name
  application_insights_id = azurerm_application_insights.azure_app_insights.id
  key_vault_id            = azurerm_key_vault.azure_kv.id
  storage_account_id      = azurerm_storage_account.azure_sa.id

  identity {
    type = "SystemAssigned"
  }
}

# ------------------------------------------------------------------------------
# Terraform Outputs
# ------------------------------------------------------------------------------
output "gcp_storage_bucket" {
  value       = google_storage_bucket.gcp_model_bucket.url
  description = "GCP Cloud Storage Bucket URL"
}

output "azure_storage_account" {
  value       = azurerm_storage_account.azure_sa.name
  description = "Azure Storage Account Name"
}

output "azure_ml_workspace" {
  value       = azurerm_machine_learning_workspace.azure_ml_workspace.name
  description = "Azure Machine Learning Workspace Name"
}
