# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # DAMO630 — Advanced Data Analytics
# MAGIC ## Final Project: E-Commerce Customer Lifecycle Intelligence Platform
# MAGIC
# MAGIC **University of Niagara Falls Canada — Master of Data Analytics**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Notebook 00 — Central Configuration
# MAGIC
# MAGIC ### Purpose
# MAGIC This notebook serves as the **single source of truth** for all project configuration.
# MAGIC Every other notebook imports this file using `%run "../00_config"`.
# MAGIC Credentials, S3 paths, Delta table names, and helper functions are all defined here.
# MAGIC
# MAGIC ### Why This Approach
# MAGIC - **Security**: Credentials are stored in one place only
# MAGIC - **Maintainability**: Change a path once — all notebooks update automatically
# MAGIC - **Reproducibility**: Any notebook can be run independently by importing this config
# MAGIC - **Professional Practice**: Mirrors industry-standard configuration management
# MAGIC
# MAGIC ### Cloud Architecture
# MAGIC - **Storage**: AWS S3 (`s3://ecom-analytics-project-2026`)
# MAGIC - **Compute**: Databricks Serverless (Apache Spark)
# MAGIC - **Auth**: Databricks External Location (no credentials in Spark reads/writes)
# MAGIC - **Catalogue**: Unity Catalog Delta Tables (`ecom_project` database)
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Library Imports
# MAGIC
# MAGIC Core Python libraries used across all notebooks.
# MAGIC These are imported once here and available to all notebooks via `%run`.

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Core libraries imported successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — AWS Credentials and Bucket Configuration
# MAGIC
# MAGIC AWS credentials are required only for reading Excel files (boto3).
# MAGIC All parquet files are read/written using Databricks External Location — no credentials needed for those.
# MAGIC
# MAGIC > **Security Note**: In a production environment these would be stored in
# MAGIC > Databricks Secrets (Key Vault). For this academic project they are stored here.

# COMMAND ----------

from dotenv import load_dotenv
import os

load_dotenv("/Workspace/Users/hsp498@gmail.com/.env")
# AWS Credentials — used for Excel file reads only
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY  = os.getenv("SECRET_KEY")
BUCKET      = os.getenv("BUCKET")
REGION      = os.getenv("REGION")

# S3 Base Path — authenticated via Databricks External Location
# External Location: db_s3_external_databricks-s3-ingest-df268
BASE_PATH = "s3://ecom-analytics-project-2026"

print(f"Bucket    : {BUCKET}")
print(f"Region    : {REGION}")
print(f"Base path : {BASE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — S3 File Path Registry
# MAGIC
# MAGIC All file paths are centralised here.
# MAGIC Use `PATHS['key']` in any notebook to reference a file.
# MAGIC
# MAGIC **Folder Structure:**
# MAGIC ```
# MAGIC s3://ecom-analytics-project-2026/
# MAGIC ├── raw/           ← Original uploaded datasets
# MAGIC ├── processed/     ← Cleaned datasets from Notebook 01
# MAGIC ├── outputs/       ← Module outputs from Notebooks 02-05
# MAGIC └── models/        ← Saved model artifacts
# MAGIC ```

# COMMAND ----------

PATHS = {
    # ── Raw data (original uploads) ───────────────────────
    "raw_retail"         : f"{BASE_PATH}/raw/online_retail_II.xlsx",
    "raw_reviews"        : f"{BASE_PATH}/raw/Reviews.csv",

    # ── Processed data (cleaned by Notebook 01) ───────────
    "retail_clean"       : f"{BASE_PATH}/processed/retail_clean.parquet",
    "reviews_clean"      : f"{BASE_PATH}/processed/reviews_clean.parquet",

    # ── Module outputs ─────────────────────────────────────
    "master_customers"   : f"{BASE_PATH}/outputs/master_customer_dataset.parquet",
    "rfm_segments"       : f"{BASE_PATH}/outputs/rfm_segments.parquet",
    "survival_data"      : f"{BASE_PATH}/outputs/customer_survival_data.parquet",
    "at_risk"            : f"{BASE_PATH}/outputs/at_risk_customers.parquet",
    "product_sentiment"  : f"{BASE_PATH}/outputs/product_sentiment.parquet",
    "recommendations"    : f"{BASE_PATH}/outputs/all_customer_recommendations.parquet",
    "reviews_sentiment"  : f"{BASE_PATH}/outputs/reviews_with_sentiment.parquet",
}

print(f"File path registry loaded — {len(PATHS)} paths configured.")
for key, path in PATHS.items():
    print(f"  {key:<25} : .../{'/'.join(path.split('/')[-2:])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — Delta Table Registry
# MAGIC
# MAGIC Delta tables are permanent cloud tables stored in Unity Catalog.
# MAGIC They are created once by Notebook 06 and can be queried from:
# MAGIC - Any notebook using `spark.sql()`
# MAGIC - The Databricks Dashboard tab (SQL interface)
# MAGIC - The Databricks SQL Editor
# MAGIC
# MAGIC **Database:** `ecom_project`

# COMMAND ----------

TABLES = {
    "master_customers"   : "ecom_project.master_customers",
    "rfm_segments"       : "ecom_project.rfm_segments",
    "survival_data"      : "ecom_project.survival_data",
    "at_risk"            : "ecom_project.at_risk_customers",
    "product_sentiment"  : "ecom_project.product_sentiment",
    "recommendations"    : "ecom_project.recommendations",
}

print(f"Delta table registry loaded — {len(TABLES)} tables configured.")
for key, table in TABLES.items():
    print(f"  {key:<25} : {table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — Helper Functions
# MAGIC
# MAGIC Reusable functions for reading and writing data.
# MAGIC These replace repetitive boto3 code in every notebook.
# MAGIC
# MAGIC | Function | Description |
# MAGIC |---|---|
# MAGIC | `read_parquet(key)` | Read parquet from S3 → pandas DataFrame |
# MAGIC | `read_parquet_spark(key)` | Read parquet from S3 → Spark DataFrame |
# MAGIC | `read_csv_spark(key)` | Read CSV from S3 → pandas DataFrame |
# MAGIC | `read_excel_boto(sheet)` | Read Excel from S3 via boto3 → pandas DataFrame |
# MAGIC | `save_parquet(df, key)` | Save pandas DataFrame → S3 parquet |
# MAGIC | `save_parquet_spark(sdf, key)` | Save Spark DataFrame → S3 parquet |
# MAGIC | `read_delta(key)` | Read Delta table → pandas DataFrame |
# MAGIC | `read_delta_spark(key)` | Read Delta table → Spark DataFrame |
# MAGIC | `list_files(prefix)` | List files in S3 folder |

# COMMAND ----------

def read_parquet(path_key):
    """
    Read a parquet file from S3 into a pandas DataFrame.
    Uses Databricks External Location — no credentials needed.

    Args:
        path_key (str): Key from PATHS dictionary

    Returns:
        pandas.DataFrame
    """
    path = PATHS[path_key]
    return spark.read.parquet(path).toPandas()


def read_parquet_spark(path_key):
    """
    Read a parquet file from S3 into a Spark DataFrame.
    Use this for very large files that should stay distributed.

    Args:
        path_key (str): Key from PATHS dictionary

    Returns:
        pyspark.sql.DataFrame
    """
    path = PATHS[path_key]
    return spark.read.parquet(path)


def read_csv_spark(path_key):
    """
    Read a CSV file from S3 into a pandas DataFrame.
    Uses Spark for distributed reading, then converts to pandas.

    Args:
        path_key (str): Key from PATHS dictionary

    Returns:
        pandas.DataFrame
    """
    path = PATHS[path_key]
    return spark.read \
               .option("header",      "true") \
               .option("inferSchema", "true") \
               .csv(path) \
               .toPandas()


def read_excel_boto(sheet_name=None):
    """
    Read the Online Retail Excel file from S3 using boto3.
    Note: Spark cannot read Excel files natively.
    boto3 is used ONLY for this Excel file.

    Args:
        sheet_name (str, optional): Excel sheet name to read

    Returns:
        pandas.DataFrame
    """
    import boto3, io
    s3  = boto3.client(
        's3',
        aws_access_key_id     = ACCESS_KEY,
        aws_secret_access_key = SECRET_KEY,
        region_name           = REGION
    )
    obj  = s3.get_object(Bucket=BUCKET, Key="raw/online_retail_II.xlsx")
    data = obj['Body'].read()
    if sheet_name:
        return pd.read_excel(
            io.BytesIO(data),
            sheet_name = sheet_name,
            engine     = "openpyxl"
        )
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


def save_parquet(df, path_key):
    """
    Save a pandas DataFrame as a parquet file to S3.
    Automatically fixes timestamp precision issues.

    Args:
        df (pandas.DataFrame): DataFrame to save
        path_key (str): Key from PATHS dictionary
    """
    path = PATHS[path_key]
    # Fix nanosecond timestamps — Databricks requires microsecond precision
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype('datetime64[us]')
    spark.createDataFrame(df) \
         .write \
         .mode("overwrite") \
         .parquet(path)
    print(f"  Saved: {path.replace(BASE_PATH, 'S3')}")


def save_parquet_spark(sdf, path_key):
    """
    Save a Spark DataFrame as a parquet file to S3.

    Args:
        sdf (pyspark.sql.DataFrame): Spark DataFrame to save
        path_key (str): Key from PATHS dictionary
    """
    path = PATHS[path_key]
    sdf.write \
       .mode("overwrite") \
       .parquet(path)
    print(f"  Saved: {path.replace(BASE_PATH, 'S3')}")


def read_delta(table_key):
    """
    Read a Delta table into a pandas DataFrame.

    Args:
        table_key (str): Key from TABLES dictionary

    Returns:
        pandas.DataFrame
    """
    table = TABLES[table_key]
    return spark.sql(f"SELECT * FROM {table}").toPandas()


def read_delta_spark(table_key):
    """
    Read a Delta table as a Spark DataFrame.

    Args:
        table_key (str): Key from TABLES dictionary

    Returns:
        pyspark.sql.DataFrame
    """
    table = TABLES[table_key]
    return spark.sql(f"SELECT * FROM {table}")


def list_files(prefix=""):
    """
    List all files in the S3 bucket under a given prefix.

    Args:
        prefix (str): Folder path to list (e.g. 'outputs/')

    Returns:
        list: File objects
    """
    path = f"{BASE_PATH}/{prefix}"
    try:
        files = dbutils.fs.ls(path)
        for f in files:
            size_mb = round(f.size / 1024 / 1024, 2)
            print(f"  {f.name:<55} {size_mb:>8.2f} MB")
        return files
    except Exception as e:
        print(f"Cannot list {path}: {e}")
        return []


print("Helper functions loaded successfully.")
print()
print("Available functions:")
print("  read_parquet(key)        — S3 parquet → pandas DataFrame")
print("  read_parquet_spark(key)  — S3 parquet → Spark DataFrame")
print("  read_csv_spark(key)      — S3 CSV     → pandas DataFrame")
print("  read_excel_boto(sheet)   — S3 Excel   → pandas DataFrame")
print("  save_parquet(df, key)    — pandas     → S3 parquet")
print("  save_parquet_spark(s,k)  — Spark      → S3 parquet")
print("  read_delta(key)          — Delta table → pandas DataFrame")
print("  read_delta_spark(key)    — Delta table → Spark DataFrame")
print("  list_files(prefix)       — List S3 files")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Display Settings

# COMMAND ----------

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows",    20)
pd.set_option("display.width",       1000)

print("Display settings configured.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Connection Verification
# MAGIC
# MAGIC Verifies that both S3 External Location and Delta tables are accessible.
# MAGIC Run this section to confirm the environment is correctly set up.

# COMMAND ----------

print("=" * 60)
print("ECOM PROJECT — Configuration Summary")
print("=" * 60)
print(f"Project     : E-Commerce Customer Lifecycle Intelligence")
print(f"Course      : DAMO630 — Advanced Data Analytics")
print(f"University  : University of Niagara Falls Canada")
print(f"Cloud       : AWS S3 + Databricks Serverless")
print(f"Bucket      : {BUCKET}")
print(f"Region      : {REGION}")
print(f"Paths       : {len(PATHS)} configured")
print(f"Tables      : {len(TABLES)} configured")
print()

# Test 1 — S3 External Location
print("Test 1 — S3 External Location Access:")
try:
    folders = dbutils.fs.ls(f"{BASE_PATH}/")
    print(f"  Status  : CONNECTED")
    print(f"  Folders : {[f.name for f in folders]}")
except Exception as e:
    print(f"  Status  : FAILED — {e}")

print()

# Test 2 — S3 File Access
print("Test 2 — S3 Files in outputs/ folder:")
try:
    files = dbutils.fs.ls(f"{BASE_PATH}/outputs/")
    print(f"  Status  : {len(files)} files found")
    for f in files:
        size_mb = round(f.size / 1024 / 1024, 2)
        print(f"    {f.name:<50} {size_mb:>6.2f} MB")
except Exception as e:
    print(f"  Status  : No output files yet — run notebooks 01-05 first")

print()

# Test 3 — Delta Tables
print("Test 3 — Delta Table Access:")
try:
    count = spark.sql(
        "SELECT COUNT(*) AS cnt FROM ecom_project.master_customers"
    ).collect()[0][0]
    print(f"  Status  : CONNECTED")
    print(f"  master_customers : {count:,} rows")
    tables = spark.sql("SHOW TABLES IN ecom_project").collect()
    print(f"  Tables available : {[t.tableName for t in tables]}")
except Exception as e:
    print(f"  Status  : Not yet created — run notebook 06_dashboard first")

print()
print("=" * 60)
print("Usage in any notebook:")
print('  %run "../00_config"')
print('  df = read_parquet("retail_clean")')
print("=" * 60)
