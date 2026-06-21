# E-Commerce Customer Lifecycle Intelligence Platform

**Course:** DAMO630 — Advanced Data Analytics  
**University:** University of Niagara Falls Canada  
**Student:** Harsh Patel
**Email:** hsp498@gmail.com  
**Date:** March 2026  

---

## Project Overview

A cloud-native analytics platform that analyses the complete e-commerce customer lifecycle using four integrated analytical modules deployed on AWS S3 and Databricks Serverless.

**Datasets:**
- UCI Online Retail II — 1,067,371 UK e-commerce transactions (2009–2011)
- Amazon Fine Food Reviews — 568,454 product reviews with star ratings

**Cloud Infrastructure:**
- AWS S3 Bucket: `ecom-analytics-project-2026` (us-east-2)
- Databricks Community Edition — Serverless Compute
- Databricks External Location connected to S3

---

## Key Results

| Module | Method | Key Finding |
|--------|--------|-------------|
| Recommendation System | Cosine Similarity CF | 29,390 recommendations for 5,878 customers |
| Survival Analysis | Kaplan-Meier + Cox PH | 50.8% churn rate · 497 at-risk customers |
| Sentiment Analysis | VADER NLP | 88.2% positive · 20,392 products scored |
| Large Scale Mining | RFM + K-Means | £17.7M revenue · 4 customer segments |

**Business Impact:**
- £306,985 recoverable through retention campaign
- £12,752,669 revenue protected in Champions segment
- 29,390 personalised recommendations deployed

---

## Repository Structure

```
ecom-recommendation-app/
├── 00_config.py                    ← Central config (credentials + helpers)
├── 01_data_preparation/
│   └── 01_data_preparation.ipynb   ← Data cleaning + EDA
├── 02_recommendation/
│   └── 02_recommendation.ipynb     ← Collaborative filtering
├── 03_survival_analysis/
│   └── 03_survival_analysis.ipynb  ← Kaplan-Meier + Cox model
├── 04_sentiment_analysis/
│   └── 04_sentiment_analysis.ipynb ← VADER sentiment NLP
├── 05_large_scale_mining/
│   └── 05_large_scale_mining.ipynb ← RFM + K-Means clustering
└── 06_dashboard/
    └── 06_dashboard.ipynb          ← Delta tables + dashboard setup
```

---

## S3 Bucket Structure

```
s3://ecom-analytics-project-2026/
├── raw/
│   ├── online_retail_II.xlsx       ← UCI Online Retail II dataset
│   └── Reviews.csv                 ← Amazon Fine Food Reviews dataset
├── processed/
│   ├── retail_clean.parquet        ← Cleaned retail data (805,549 rows)
│   └── reviews_clean.parquet       ← Cleaned reviews (560,777 rows)
├── outputs/
│   ├── all_customer_recommendations.parquet
│   ├── customer_survival_data.parquet
│   ├── at_risk_customers.parquet
│   ├── reviews_with_sentiment.parquet
│   ├── product_sentiment.parquet
│   ├── master_customer_dataset.parquet
│   ├── rfm_segments.parquet
│   ├── results_recommendation_model.json
│   ├── results_survival_analysis.json
│   ├── results_sentiment_analysis.json
│   └── results_large_scale_mining.json
└── models/
    └── customer_similarity_matrix.parquet
```

---

## How to Run

### Prerequisites

1. Databricks Community Edition account
2. AWS account with S3 bucket `ecom-analytics-project-2026`
3. Both datasets uploaded to `s3://ecom-analytics-project-2026/raw/`
4. Databricks External Location connected to the S3 bucket

### Credentials Setup

Create a `.env` file in your Databricks workspace:

```
/Workspace/Users/your-email/ecom-recommendation-app/.env
```

Contents:
```
ACCESS_KEY=YOUR_AWS_ACCESS_KEY
SECRET_KEY=YOUR_AWS_SECRET_KEY
```

### Run Order

Run notebooks in this exact order. Each notebook depends on the outputs of the previous one.

```
Step 1 → 01_data_preparation    Cleans raw data → saves to processed/
Step 2 → 02_recommendation      Builds CF model → saves recommendations
Step 3 → 03_survival_analysis   Fits KM + Cox  → saves survival data
Step 4 → 04_sentiment_analysis  Runs VADER     → saves sentiment scores
Step 5 → 05_large_scale_mining  RFM + K-Means  → saves master dataset
Step 6 → 06_dashboard           Creates Delta tables → powers dashboard
```

### Running a Notebook

1. Open notebook in Databricks workspace
2. Click **Run All** at the top
3. Wait for **"COMPLETE"** message in the last cell
4. Proceed to the next notebook

---

## Libraries Required

Each notebook installs its own dependencies via `%pip install`. No manual installation needed.

| Notebook | Key Libraries |
|----------|--------------|
| 01_data_preparation | openpyxl, boto3, pandas, matplotlib, seaborn |
| 02_recommendation | scikit-learn, scipy, pandas, numpy, boto3 |
| 03_survival_analysis | lifelines, pandas, numpy, matplotlib, boto3 |
| 04_sentiment_analysis | vaderSentiment, wordcloud, pandas, matplotlib, boto3 |
| 05_large_scale_mining | scikit-learn, pandas, numpy, matplotlib, boto3 |
| 06_dashboard | pandas, boto3, pyspark (built-in) |

---

## Dashboard

The Databricks Dashboard is powered by 6 Delta tables in the `ecom_project` database:

| Table | Rows | Description |
|-------|------|-------------|
| ecom_project.master_customers | 5,878 | All customer features combined |
| ecom_project.rfm_segments | 5,878 | RFM scores and segments |
| ecom_project.survival_data | 5,878 | Churn and survival features |
| ecom_project.at_risk_customers | 497 | At-risk retention targets |
| ecom_project.product_sentiment | 20,392 | Product sentiment scores |
| ecom_project.recommendations | 29,390 | Customer recommendations |

**To recreate Delta tables:** Run notebook `06_dashboard` — it loads all parquet files from S3 and saves them as permanent Delta tables.

---

## Credential Security

AWS credentials are stored in a `.env` file using `python-dotenv` and are never hardcoded in any notebook file. The `.env` file is excluded from this submission.

---

## Known Limitations

- **MLflow:** Databricks Serverless Community Edition does not support MLflow Model Registry URI. Model results are saved as JSON files to S3 as an alternative.
- **Excel reading:** Apache Spark cannot read `.xlsx` files natively. `boto3` is used in Notebook 01 only for the Excel file.
- **Serverless restrictions:** `sc._jsc.hadoopConfiguration()` is not available on Serverless. All S3 reads use `boto3` or the Databricks External Location.

---

## Datasets

| Dataset | Source | Link |
|---------|--------|------|
| Online Retail II | UCI ML Repository | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Amazon Fine Food Reviews | Kaggle / Stanford | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews |

---

## Contact

**Harsh Patel**  
hsp498@gmail.com  
University of Niagara Falls Canada  
DAMO630 — Advanced Data Analytics
