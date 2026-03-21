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
# MAGIC ## Notebook 01 — Data Preparation and Exploratory Data Analysis
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook is the **first stage** of the analytics pipeline.
# MAGIC It loads two real-world datasets from AWS S3, cleans and validates them,
# MAGIC performs exploratory data analysis, and saves the cleaned datasets back to S3
# MAGIC for use by all subsequent analytical modules.
# MAGIC
# MAGIC ### Datasets
# MAGIC | Dataset | Source | Size | Description |
# MAGIC |---|---|---|---|
# MAGIC | Online Retail II | UCI ML Repository | 1,067,371 rows | UK e-commerce transactions 2009–2011 |
# MAGIC | Amazon Fine Food Reviews | Kaggle / Snap | 568,454 rows | Amazon product reviews with ratings |
# MAGIC
# MAGIC ### Objectives
# MAGIC 1. Load both raw datasets from S3 into Databricks
# MAGIC 2. Identify and handle missing values, duplicates, and data quality issues
# MAGIC 3. Engineer new features required by downstream modules
# MAGIC 4. Conduct exploratory data analysis with business-oriented visualisations
# MAGIC 5. Save cleaned datasets to S3 `processed/` folder
# MAGIC
# MAGIC ### Outputs
# MAGIC - `s3://ecom-analytics-project-2026/processed/retail_clean.parquet`
# MAGIC - `s3://ecom-analytics-project-2026/processed/reviews_clean.parquet`
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Install Required Libraries
# MAGIC
# MAGIC Installing all libraries required for data loading, processing, and visualisation.
# MAGIC `openpyxl` is required specifically for reading Excel files.
# MAGIC `boto3` is required for reading the Excel file from S3 (Spark cannot read Excel natively).

# COMMAND ----------

%pip install openpyxl pandas numpy matplotlib seaborn pyarrow boto3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — Load Central Configuration
# MAGIC
# MAGIC Importing credentials, S3 paths, and helper functions from the central config notebook.
# MAGIC This ensures consistency across all project notebooks.

# COMMAND ----------

%run "../00_config"

import matplotlib.pyplot as plt
import seaborn as sns

print("Configuration loaded successfully!")
print(f"Base path : {BASE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — Load Raw Datasets from AWS S3
# MAGIC
# MAGIC Loading both raw datasets directly from the S3 `raw/` folder.
# MAGIC
# MAGIC **Online Retail II** is an Excel file with two worksheets:
# MAGIC - `Year 2009-2010`: Transactions from December 2009 to December 2010
# MAGIC - `Year 2010-2011`: Transactions from December 2010 to December 2011
# MAGIC
# MAGIC Both sheets are combined into a single DataFrame for analysis.
# MAGIC
# MAGIC **Amazon Fine Food Reviews** is a CSV file containing
# MAGIC product reviews, star ratings, and reviewer metadata.

# COMMAND ----------

print("=" * 60)
print("Loading Raw Datasets from AWS S3")
print("=" * 60)

# ── Online Retail II — Excel file ─────────────────────────
# Note: Apache Spark cannot read Excel files natively.
# boto3 is used here — the only place in the project where boto3 is needed.
print("\nStep 1: Loading Online Retail II (Excel)...")
print("  This may take 1-2 minutes for the large Excel file...")

df_2009   = read_excel_boto(sheet_name="Year 2009-2010")
df_2010   = read_excel_boto(sheet_name="Year 2010-2011")
retail_df = pd.concat([df_2009, df_2010], ignore_index=True)

# Fix mixed type columns — StockCode contains both numeric and string values
retail_df['StockCode'] = retail_df['StockCode'].astype(str)
retail_df['Invoice']   = retail_df['Invoice'].astype(str)

print(f"\n  Online Retail II loaded successfully:")
print(f"  Total rows          : {len(retail_df):,}")
print(f"  Total columns       : {len(retail_df.columns)}")
print(f"  Columns             : {list(retail_df.columns)}")
print(f"  Date range          : {retail_df['InvoiceDate'].min()} → {retail_df['InvoiceDate'].max()}")
print(f"  Unique customers    : {retail_df['Customer ID'].nunique():,}")
print(f"  Unique products     : {retail_df['StockCode'].nunique():,}")
print(f"  Unique countries    : {retail_df['Country'].nunique():,}")

# ── Amazon Fine Food Reviews — CSV file ───────────────────
print("\nStep 2: Loading Amazon Fine Food Reviews (CSV)...")
reviews_df = read_csv_spark("raw_reviews")

print(f"\n  Amazon Reviews loaded successfully:")
print(f"  Total rows          : {len(reviews_df):,}")
print(f"  Total columns       : {len(reviews_df.columns)}")
print(f"  Columns             : {list(reviews_df.columns)}")
print(f"  Unique products     : {reviews_df['ProductId'].nunique():,}")
print(f"  Unique reviewers    : {reviews_df['UserId'].nunique():,}")
print(f"  Rating range        : {reviews_df['Score'].min()} to {reviews_df['Score'].max()} stars")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — Data Quality Assessment
# MAGIC
# MAGIC Before cleaning, we assess the data quality of both datasets by:
# MAGIC - Identifying columns with missing values
# MAGIC - Calculating the percentage of missing data per column
# MAGIC - Understanding the business impact of missing values
# MAGIC
# MAGIC **Key Finding (Online Retail):**
# MAGIC Customer ID is missing for 22.77% of rows — these represent guest checkout
# MAGIC transactions where no account was created. These rows cannot be used for
# MAGIC customer-level analysis and will be removed.

# COMMAND ----------

print("=" * 60)
print("Data Quality Assessment — Missing Values")
print("=" * 60)

# Online Retail missing values
print("\nOnline Retail II — Missing Value Analysis:")
retail_missing     = retail_df.isnull().sum()
retail_missing_pct = (retail_df.isnull().sum() / len(retail_df) * 100).round(2)
retail_missing_df  = pd.DataFrame({
    'Column'        : retail_missing.index,
    'Missing Count' : retail_missing.values,
    'Missing %'     : retail_missing_pct.values
})
retail_missing_df  = retail_missing_df[retail_missing_df['Missing Count'] > 0]
print(retail_missing_df.to_string(index=False))

print(f"\nBusiness Impact:")
print(f"  Customer ID missing = {retail_df['Customer ID'].isna().sum():,} rows ({retail_df['Customer ID'].isna().mean()*100:.1f}%)")
print(f"  These are guest checkout transactions — cannot track customer behaviour")
print(f"  Action: Remove rows with missing Customer ID")

# Amazon Reviews missing values
print("\n\nAmazon Reviews — Missing Value Analysis:")
reviews_missing     = reviews_df.isnull().sum()
reviews_missing_pct = (reviews_df.isnull().sum() / len(reviews_df) * 100).round(2)
reviews_missing_df  = pd.DataFrame({
    'Column'        : reviews_missing.index,
    'Missing Count' : reviews_missing.values,
    'Missing %'     : reviews_missing_pct.values
})
reviews_missing_df  = reviews_missing_df[reviews_missing_df['Missing Count'] > 0]
if len(reviews_missing_df) > 0:
    print(reviews_missing_df.to_string(index=False))
else:
    print("  No missing values found — dataset is complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — Clean Online Retail Dataset
# MAGIC
# MAGIC Applying the following cleaning steps in order:
# MAGIC
# MAGIC | Step | Action | Reason |
# MAGIC |---|---|---|
# MAGIC | 1 | Remove null Customer ID | Cannot perform customer analysis without ID |
# MAGIC | 2 | Remove cancelled orders (Invoice starts with 'C') | Cancellations are not purchases |
# MAGIC | 3 | Remove zero or negative quantities | Data entry errors or returns |
# MAGIC | 4 | Remove zero or negative prices | Data entry errors or free items |
# MAGIC | 5 | Convert Customer ID to integer | Remove decimal formatting |
# MAGIC | 6 | Create TotalSpend column | Required for RFM and survival analysis |
# MAGIC | 7 | Parse InvoiceDate as datetime | Required for time-series analysis |
# MAGIC | 8 | Extract Year, Month, Day features | Required for trend analysis |

# COMMAND ----------

print("Cleaning Online Retail Dataset...")
print(f"Rows before cleaning : {len(retail_df):,}")
print()

# Step 1 — Remove rows with missing Customer ID
retail_clean = retail_df.dropna(subset=['Customer ID'])
print(f"Step 1 — Remove null Customer ID     : {len(retail_clean):,} rows remaining")

# Step 2 — Remove cancelled orders
retail_clean = retail_clean[~retail_clean['Invoice'].str.startswith('C')]
print(f"Step 2 — Remove cancelled orders     : {len(retail_clean):,} rows remaining")

# Step 3 — Remove negative or zero quantities
retail_clean = retail_clean[retail_clean['Quantity'] > 0]
print(f"Step 3 — Remove invalid quantities   : {len(retail_clean):,} rows remaining")

# Step 4 — Remove negative or zero prices
retail_clean = retail_clean[retail_clean['Price'] > 0]
print(f"Step 4 — Remove invalid prices       : {len(retail_clean):,} rows remaining")

# Step 5 — Convert Customer ID to integer
retail_clean['Customer ID'] = retail_clean['Customer ID'].astype(int)
print(f"Step 5 — Convert Customer ID to int  : Done")

# Step 6 — Create TotalSpend (Quantity × Price)
retail_clean['TotalSpend'] = retail_clean['Quantity'] * retail_clean['Price']
print(f"Step 6 — Create TotalSpend column    : Done (max = £{retail_clean['TotalSpend'].max():,.2f})")

# Step 7 — Parse InvoiceDate as datetime
retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])
print(f"Step 7 — Parse InvoiceDate           : Done")

# Step 8 — Extract date features
retail_clean['Year']  = retail_clean['InvoiceDate'].dt.year
retail_clean['Month'] = retail_clean['InvoiceDate'].dt.month
retail_clean['Day']   = retail_clean['InvoiceDate'].dt.day_name()
print(f"Step 8 — Extract date features       : Done (Year, Month, Day added)")

print()
print(f"Final clean rows     : {len(retail_clean):,}")
print(f"Rows removed total   : {len(retail_df) - len(retail_clean):,} ({(len(retail_df) - len(retail_clean))/len(retail_df)*100:.1f}%)")
print()
print("Sample of cleaned data:")
print(retail_clean.head(5).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Clean Amazon Reviews Dataset
# MAGIC
# MAGIC Applying the following cleaning steps:
# MAGIC
# MAGIC | Step | Action | Reason |
# MAGIC |---|---|---|
# MAGIC | 1 | Remove rows with missing text | Cannot analyse sentiment without text |
# MAGIC | 2 | Remove duplicate reviews (same user + product) | Prevent bias in sentiment scores |
# MAGIC | 3 | Keep only required columns | Reduce memory usage |
# MAGIC | 4 | Convert Unix timestamp to readable date | Required for trend analysis |
# MAGIC | 5 | Rename Score to Rating | Improve clarity |
# MAGIC | 6 | Add ReviewLength feature | Longer reviews may have stronger sentiment |

# COMMAND ----------

print("Cleaning Amazon Reviews Dataset...")
print(f"Rows before cleaning : {len(reviews_df):,}")
print()

# Step 1 — Remove rows with missing review text
reviews_clean = reviews_df.dropna(subset=['Text', 'Summary'])
print(f"Step 1 — Remove null text/summary    : {len(reviews_clean):,} rows remaining")

# Step 2 — Remove duplicate reviews
reviews_clean = reviews_clean.drop_duplicates(
    subset=['UserId', 'ProductId'], keep='first'
)
print(f"Step 2 — Remove duplicate reviews    : {len(reviews_clean):,} rows remaining")

# Step 3 — Keep required columns only
reviews_clean = reviews_clean[
    ['ProductId', 'UserId', 'Score', 'Summary', 'Text', 'Time']
].copy()
print(f"Step 3 — Keep required columns       : {list(reviews_clean.columns)}")

# Step 4 — Convert Unix timestamp to datetime
reviews_clean['ReviewDate'] = pd.to_datetime(
    reviews_clean['Time'], unit='s'
).astype('datetime64[us]')
print(f"Step 4 — Convert Unix timestamp      : Done")

# Step 5 — Rename Score to Rating
reviews_clean.rename(columns={'Score': 'Rating'}, inplace=True)
print(f"Step 5 — Rename Score → Rating       : Done")

# Step 6 — Add review length feature
reviews_clean['ReviewLength'] = reviews_clean['Text'].apply(len)
print(f"Step 6 — Add ReviewLength feature    : Done (avg = {reviews_clean['ReviewLength'].mean():.0f} chars)")

print()
print(f"Final clean rows     : {len(reviews_clean):,}")
print(f"Rows removed total   : {len(reviews_df) - len(reviews_clean):,}")
print()
print("Sample of cleaned data:")
print(reviews_clean.head(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Exploratory Data Analysis
# MAGIC
# MAGIC Four business-oriented visualisations are generated to understand
# MAGIC the key patterns in the data before modelling begins.
# MAGIC
# MAGIC **Chart 1 — Monthly Revenue Trend**
# MAGIC Shows seasonal patterns and overall revenue trajectory.
# MAGIC
# MAGIC **Chart 2 — Top 10 Countries by Revenue**
# MAGIC Reveals geographic concentration of the customer base.
# MAGIC
# MAGIC **Chart 3 — Amazon Review Rating Distribution**
# MAGIC Shows the distribution of customer satisfaction across star ratings.
# MAGIC
# MAGIC **Chart 4 — Orders by Day of Week**
# MAGIC Reveals which days drive the most transaction volume.

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'E-Commerce Customer Lifecycle — Exploratory Data Analysis',
    fontsize=16, fontweight='bold', y=1.02
)

# ── Chart 1 — Monthly Revenue Trend ───────────────────────
monthly = retail_clean.groupby(
    retail_clean['InvoiceDate'].dt.to_period('M')
)['TotalSpend'].sum().reset_index()
monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)

axes[0,0].plot(monthly['InvoiceDate'], monthly['TotalSpend'],
               color='#2196F3', linewidth=2, marker='o', markersize=4)
axes[0,0].set_title('Monthly Revenue Trend', fontweight='bold', fontsize=12)
axes[0,0].set_xlabel('Month')
axes[0,0].set_ylabel('Total Revenue (£)')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_facecolor('#FAFAFA')

# ── Chart 2 — Top 10 Countries by Revenue ─────────────────
top_countries = retail_clean.groupby('Country')['TotalSpend'] \
    .sum().sort_values(ascending=False).head(10)

axes[0,1].barh(top_countries.index[::-1], top_countries.values[::-1],
               color='#4CAF50', edgecolor='white', linewidth=1)
axes[0,1].set_title('Top 10 Countries by Revenue', fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('Total Revenue (£)')
axes[0,1].set_facecolor('#FAFAFA')

# ── Chart 3 — Rating Distribution ─────────────────────────
rating_counts = reviews_clean['Rating'].value_counts().sort_index()
colors        = ['#F44336','#FF9800','#FFC107','#8BC34A','#4CAF50']

axes[1,0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='white')
axes[1,0].set_title('Amazon Review Rating Distribution', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('Star Rating')
axes[1,0].set_ylabel('Number of Reviews')
for idx, val in rating_counts.items():
    axes[1,0].text(idx, val + 500, f'{val:,}', ha='center', fontsize=9)
axes[1,0].set_facecolor('#FAFAFA')

# ── Chart 4 — Orders by Day of Week ───────────────────────
day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_counts = retail_clean['Day'].value_counts().reindex(day_order)

axes[1,1].bar(day_counts.index, day_counts.values, color='#9C27B0', edgecolor='white')
axes[1,1].set_title('Orders by Day of Week', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Day of Week')
axes[1,1].set_ylabel('Number of Orders')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('/tmp/eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()

print("EDA Key Findings:")
print(f"  Peak revenue month    : {monthly.loc[monthly['TotalSpend'].idxmax(), 'InvoiceDate']}")
print(f"  Top country           : {top_countries.index[0]} (£{top_countries.values[0]:,.0f})")
print(f"  Most popular rating   : {rating_counts.idxmax()} stars ({rating_counts.max():,} reviews)")
print(f"  Busiest day           : {day_counts.idxmax()} ({day_counts.max():,} orders)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 8 — Save Cleaned Datasets to S3
# MAGIC
# MAGIC Saving both cleaned datasets to the S3 `processed/` folder as Parquet files.
# MAGIC Parquet format is chosen because:
# MAGIC - **Columnar storage**: Much faster for analytical queries than CSV
# MAGIC - **Compression**: Significantly smaller file sizes
# MAGIC - **Schema preservation**: Data types are maintained automatically
# MAGIC - **Spark native**: Optimal for Spark-based processing in subsequent notebooks

# COMMAND ----------

print("Saving cleaned datasets to S3 processed/ folder...")
print()

# Save Online Retail cleaned data
print("Saving retail_clean.parquet...")
save_parquet(retail_clean, "retail_clean")

# Save Amazon Reviews cleaned data
print("Saving reviews_clean.parquet...")
save_parquet(reviews_clean, "reviews_clean")

# Verify files were saved
print()
print("Verifying files in S3 processed/ folder:")
list_files("processed/")

print()
print("=" * 60)
print("NOTEBOOK 01 — DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"Online Retail II clean : {len(retail_clean):,} rows saved")
print(f"Amazon Reviews clean   : {len(reviews_clean):,} rows saved")
print()
print("Key Statistics:")
print(f"  Date range           : {retail_clean['InvoiceDate'].min().date()} → {retail_clean['InvoiceDate'].max().date()}")
print(f"  Unique customers     : {retail_clean['Customer ID'].nunique():,}")
print(f"  Unique products      : {retail_clean['StockCode'].nunique():,}")
print(f"  Unique countries     : {retail_clean['Country'].nunique():,}")
print(f"  Total revenue        : £{retail_clean['TotalSpend'].sum():,.2f}")
print()
print("Next Step: Run Notebook 02 — Recommendation System")
