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
# MAGIC ## Notebook 06 — Dashboard Setup and Business Visualisation
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook creates the **executive dashboard** for the project.
# MAGIC It performs two functions:
# MAGIC
# MAGIC 1. **Delta Table Creation**: Converts all S3 parquet outputs into
# MAGIC    permanent Delta tables in Unity Catalog. These tables power the
# MAGIC    Databricks Dashboard tab with live SQL queries.
# MAGIC
# MAGIC 2. **Matplotlib Dashboard**: Generates a professional two-page
# MAGIC    static dashboard suitable for inclusion in the technical report
# MAGIC    and business presentation.
# MAGIC
# MAGIC ### Why Delta Tables for the Dashboard?
# MAGIC Unlike temporary views (which exist only in a notebook session),
# MAGIC Delta tables persist permanently in Unity Catalog and are accessible from:
# MAGIC - The Databricks Dashboard tab (interactive SQL dashboard)
# MAGIC - The SQL Editor
# MAGIC - Any notebook in the workspace
# MAGIC - External BI tools via JDBC/ODBC
# MAGIC
# MAGIC ### Dashboard Content
# MAGIC The dashboard presents insights from all four analytical modules:
# MAGIC
# MAGIC | Panel | Content | Source Module |
# MAGIC |---|---|---|
# MAGIC | KPI 1 | Total Customers (5,878) | Module 4 |
# MAGIC | KPI 2 | Total Revenue (£17.7M) | Module 4 |
# MAGIC | KPI 3 | Churn Rate (50.8%) | Module 2 |
# MAGIC | KPI 4 | At-Risk Customers (497) | Module 2 |
# MAGIC | Chart 1 | Customer Segment Distribution | Module 4 |
# MAGIC | Chart 2 | Revenue by Segment | Module 4 |
# MAGIC | Chart 3 | Churn Rate by Segment | Module 2 |
# MAGIC | Chart 4 | Review Sentiment Distribution | Module 3 |
# MAGIC | Chart 5 | Top 15 Recommended Products | Module 1 |
# MAGIC | Chart 6 | RFM Score Distribution | Module 4 |
# MAGIC
# MAGIC ### Run Order
# MAGIC This notebook must be run **after** all five analysis notebooks
# MAGIC (01 through 05) have been run successfully and all output files
# MAGIC are present in S3.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Install Required Libraries

# COMMAND ----------

%pip install pandas numpy matplotlib seaborn pyarrow boto3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — Load Configuration

# COMMAND ----------

%run "../00_config"

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

print("Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — Create Permanent Delta Tables
# MAGIC
# MAGIC Converting all S3 parquet output files into permanent Delta tables
# MAGIC stored in the `ecom_project` database in Unity Catalog.
# MAGIC
# MAGIC **Timestamp Fix:**
# MAGIC Pandas stores timestamps in nanosecond precision (`datetime64[ns]`).
# MAGIC Databricks/Delta requires microsecond precision (`datetime64[us]`).
# MAGIC All datetime columns are converted before saving as Delta tables.
# MAGIC
# MAGIC **Run this section once.** Delta tables persist permanently —
# MAGIC you do not need to re-run this after the tables are created.

# COMMAND ----------

print("Creating permanent Delta tables in Unity Catalog...")
print("Database: ecom_project")
print()

# Create database if it does not exist
spark.sql("CREATE DATABASE IF NOT EXISTS ecom_project")
print("Database 'ecom_project' ready.")
print()

def fix_timestamps_and_save_delta(path_key, table_name):
    """
    Load parquet from S3, fix timestamp precision,
    and save as a permanent Delta table.

    Args:
        path_key (str): Key from PATHS dictionary
        table_name (str): Delta table name (without database prefix)
    """
    path = PATHS[path_key]
    print(f"Processing {table_name}...")

    # Step 1 — Load from S3
    pdf = spark.read.parquet(path).toPandas()

    # Step 2 — Clean column names (remove spaces and hyphens)
    pdf.columns = [c.replace(' ', '_').replace('-', '_') for c in pdf.columns]

    # Step 3 — Fix nanosecond timestamps → microseconds
    for col in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[col]):
            pdf[col] = pdf[col].astype('datetime64[us]')

    # Step 4 — Convert to Spark DataFrame
    sdf = spark.createDataFrame(pdf)

    # Step 5 — Cast any remaining timestamp columns
    for field in sdf.schema.fields:
        if 'timestamp' in str(field.dataType).lower():
            sdf = sdf.withColumn(field.name, F.col(field.name).cast(TimestampType()))

    # Step 6 — Save as Delta table
    sdf.write \
       .format("delta") \
       .mode("overwrite") \
       .saveAsTable(f"ecom_project.{table_name}")

    count = spark.sql(
        f"SELECT COUNT(*) AS cnt FROM ecom_project.{table_name}"
    ).collect()[0][0]
    print(f"  Saved: ecom_project.{table_name} ({count:,} rows)")
    return count


# Create all 6 Delta tables
fix_timestamps_and_save_delta("master_customers",  "master_customers")
fix_timestamps_and_save_delta("rfm_segments",      "rfm_segments")
fix_timestamps_and_save_delta("survival_data",     "survival_data")
fix_timestamps_and_save_delta("at_risk",           "at_risk_customers")
fix_timestamps_and_save_delta("product_sentiment", "product_sentiment")
fix_timestamps_and_save_delta("recommendations",   "recommendations")

print()
print("=== All Delta Tables Created Successfully ===")
spark.sql("SHOW TABLES IN ecom_project").show(truncate=False)

# Verification query
print("Verification query — Revenue by segment:")
spark.sql("""
    SELECT
        RFM_Segment                        AS Segment,
        COUNT(*)                           AS Customers,
        ROUND(SUM(Monetary), 0)            AS Total_Revenue,
        ROUND(AVG(churned) * 100, 1)       AS Churn_Rate_Pct
    FROM ecom_project.master_customers
    GROUP BY RFM_Segment
    ORDER BY Total_Revenue DESC
""").show()

print("Delta tables ready for Dashboard tab!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — Dashboard SQL Query Reference
# MAGIC
# MAGIC Use these SQL queries in the Databricks Dashboard tab.
# MAGIC Go to **Dashboards** in the left sidebar → Create dashboard → Add visualization → paste these queries.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### KPI 1 — Total Customers
# MAGIC ```sql
# MAGIC SELECT COUNT(*) AS Total_Customers
# MAGIC FROM ecom_project.master_customers
# MAGIC ```
# MAGIC
# MAGIC #### KPI 2 — Total Revenue
# MAGIC ```sql
# MAGIC SELECT ROUND(SUM(Monetary), 2) AS Total_Revenue_GBP
# MAGIC FROM ecom_project.master_customers
# MAGIC ```
# MAGIC
# MAGIC #### KPI 3 — Overall Churn Rate
# MAGIC ```sql
# MAGIC SELECT ROUND(AVG(churned) * 100, 1) AS Churn_Rate_Pct
# MAGIC FROM ecom_project.master_customers
# MAGIC ```
# MAGIC
# MAGIC #### KPI 4 — At-Risk Customers
# MAGIC ```sql
# MAGIC SELECT COUNT(*) AS At_Risk_Customers
# MAGIC FROM ecom_project.at_risk_customers
# MAGIC ```
# MAGIC
# MAGIC #### Chart — Customer Segment Distribution (Pie)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     RFM_Segment AS Segment,
# MAGIC     COUNT(*) AS Customer_Count,
# MAGIC     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS Percentage
# MAGIC FROM ecom_project.rfm_segments
# MAGIC GROUP BY RFM_Segment
# MAGIC ORDER BY Customer_Count DESC
# MAGIC ```
# MAGIC
# MAGIC #### Chart — Revenue by Segment (Bar)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     RFM_Segment AS Segment,
# MAGIC     ROUND(SUM(Monetary), 2) AS Total_Revenue,
# MAGIC     COUNT(*) AS Customers,
# MAGIC     ROUND(SUM(Monetary) * 100.0 / SUM(SUM(Monetary)) OVER(), 1) AS Revenue_Share_Pct
# MAGIC FROM ecom_project.master_customers
# MAGIC GROUP BY RFM_Segment
# MAGIC ORDER BY Total_Revenue DESC
# MAGIC ```
# MAGIC
# MAGIC #### Chart — Churn Rate by Segment (Bar)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     RFM_Segment AS Segment,
# MAGIC     ROUND(AVG(churned) * 100, 1) AS Churn_Rate_Pct,
# MAGIC     COUNT(*) AS Total_Customers,
# MAGIC     SUM(churned) AS Churned_Count
# MAGIC FROM ecom_project.master_customers
# MAGIC GROUP BY RFM_Segment
# MAGIC ORDER BY Churn_Rate_Pct DESC
# MAGIC ```
# MAGIC
# MAGIC #### Chart — Top 15 Most Recommended Products (Horizontal Bar)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     Description AS Product,
# MAGIC     ROUND(SUM(Score), 0) AS Total_Score,
# MAGIC     COUNT(DISTINCT CustomerID) AS Recommended_To_Customers
# MAGIC FROM ecom_project.recommendations
# MAGIC GROUP BY Description
# MAGIC ORDER BY Total_Score DESC
# MAGIC LIMIT 15
# MAGIC ```
# MAGIC
# MAGIC #### Chart — Product Sentiment Distribution (Bar)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     CASE
# MAGIC         WHEN avg_sentiment >= 0.5    THEN '5 - Very Positive (0.5 to 1.0)'
# MAGIC         WHEN avg_sentiment >= 0.05   THEN '4 - Positive (0.05 to 0.5)'
# MAGIC         WHEN avg_sentiment >= -0.05  THEN '3 - Neutral'
# MAGIC         WHEN avg_sentiment >= -0.5   THEN '2 - Negative (-0.5 to -0.05)'
# MAGIC         ELSE                              '1 - Very Negative (below -0.5)'
# MAGIC     END AS Sentiment_Band,
# MAGIC     COUNT(*) AS Product_Count,
# MAGIC     ROUND(AVG(avg_rating), 2) AS Avg_Star_Rating
# MAGIC FROM ecom_project.product_sentiment
# MAGIC GROUP BY Sentiment_Band
# MAGIC ORDER BY Sentiment_Band
# MAGIC ```
# MAGIC
# MAGIC #### Table — At-Risk Customers (Table visual)
# MAGIC ```sql
# MAGIC SELECT
# MAGIC     CAST(Customer_ID AS STRING) AS Customer_ID,
# MAGIC     days_since_last AS Days_Inactive,
# MAGIC     total_orders AS Total_Orders,
# MAGIC     ROUND(total_spend, 2) AS Historical_Spend_GBP,
# MAGIC     ROUND(risk_score, 4) AS Risk_Score,
# MAGIC     segment AS Value_Segment
# MAGIC FROM ecom_project.at_risk_customers
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 50
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Dashboard Layout:**
# MAGIC ```
# MAGIC Row 1: KPI1    KPI2    KPI3    KPI4
# MAGIC Row 2: Segment Pie     Revenue Bar
# MAGIC Row 3: Churn Bar       Sentiment Bar
# MAGIC Row 4: Top Products (full width)
# MAGIC Row 5: At-Risk Table (full width)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — Load All Data for Matplotlib Dashboard

# COMMAND ----------

print("Loading all output data for matplotlib dashboard...")
print()

master    = read_parquet("master_customers")
rfm       = read_parquet("rfm_segments")
survival  = read_parquet("survival_data")
at_risk   = read_parquet("at_risk")
prod_sent = read_parquet("product_sentiment")
recs      = read_parquet("recommendations")

print(f"  Master customers  : {len(master):,} rows")
print(f"  RFM segments      : {len(rfm):,} rows")
print(f"  Survival data     : {len(survival):,} rows")
print(f"  At-risk customers : {len(at_risk):,} rows")
print(f"  Product sentiment : {len(prod_sent):,} rows")
print(f"  Recommendations   : {len(recs):,} rows")
print()
print("All data loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Dashboard Page 1: Executive Summary
# MAGIC
# MAGIC Six-panel executive summary showing the key business metrics
# MAGIC and analytical insights from all four modules.
# MAGIC Designed for a non-technical C-suite audience.

# COMMAND ----------

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35,
                        top=0.88, bottom=0.08, left=0.06, right=0.97)

# ── Dashboard Title ────────────────────────────────────────
fig.text(0.5, 0.95,
         'E-Commerce Customer Lifecycle Intelligence Dashboard',
         ha='center', fontsize=22, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.915,
         'University of Niagara Falls Canada  |  DAMO630  |  AWS S3 + Databricks  |  4 Modules  |  5,878 Customers  |  £17.7M Revenue',
         ha='center', fontsize=11, color='#555555')
fig.add_artist(plt.Line2D([0.05, 0.95], [0.905, 0.905],
               color='#1D9E75', linewidth=2.5, transform=fig.transFigure))

# ── Colour scheme ──────────────────────────────────────────
colors_seg = {
    'Champions'      : '#1D9E75',
    'Loyal Customers': '#2196F3',
    'At Risk'        : '#F39C12',
    'Lost'           : '#E74C3C'
}

# ── KPI Cards (row 1) ──────────────────────────────────────
kpi_data = [
    (f"{len(master):,}",
     'Total Customers',
     'Unique buyers',
     '#1D9E75', '#E8F8F5'),

    (f"£{master['Monetary'].sum()/1e6:.1f}M",
     'Total Revenue',
     'All customer segments',
     '#2196F3', '#E3F2FD'),

    (f"{master['churned'].mean()*100:.1f}%",
     'Overall Churn Rate',
     f"{master['churned'].sum():,} customers lost",
     '#E74C3C', '#FDEDEC'),

    (f"{len(at_risk):,}",
     'At-Risk Customers',
     '£306K recoverable revenue',
     '#F39C12', '#FEF9E7'),
]

for i, (value, title, subtitle, color, bg) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.62, value, ha='center', va='center',
            transform=ax.transAxes, fontsize=26, fontweight='bold', color=color)
    ax.text(0.5, 0.32, title, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(0.5, 0.12, subtitle, ha='center', va='center',
            transform=ax.transAxes, fontsize=9, color='#7f8c8d')

# ── Chart 1 — Customer Segment Donut ──────────────────────
ax1      = fig.add_subplot(gs[1, 0])
seg_data = rfm['RFM_Segment'].value_counts()
ax1.pie(
    seg_data.values,
    labels     = [f"{s}\n({c:,})" for s, c in zip(seg_data.index, seg_data.values)],
    colors     = [colors_seg.get(s,'#999') for s in seg_data.index],
    autopct    = '%1.1f%%',
    startangle = 90,
    pctdistance= 0.76,
    wedgeprops = dict(width=0.55, edgecolor='white', linewidth=2)
)
ax1.set_title('Customer Segments\n(Module 4 — RFM Analysis)', fontweight='bold', fontsize=10)

# ── Chart 2 — Revenue by Segment ──────────────────────────
ax2         = fig.add_subplot(gs[1, 1])
seg_revenue = master.groupby('RFM_Segment')['Monetary'].sum().sort_values(ascending=False)
bars = ax2.bar(
    range(len(seg_revenue)), seg_revenue.values/1e6,
    color     = [colors_seg.get(s,'#999') for s in seg_revenue.index],
    edgecolor = 'white', linewidth=1.5, width=0.6
)
ax2.set_xticks(range(len(seg_revenue)))
ax2.set_xticklabels([s.replace(' ','\n') for s in seg_revenue.index], fontsize=8)
ax2.set_title('Revenue by Segment\n(Module 4)', fontweight='bold', fontsize=10)
ax2.set_ylabel('Revenue (£M)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:.1f}M'))
ax2.set_facecolor('#FAFAFA')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, seg_revenue.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
             f'£{val/1e6:.1f}M', ha='center', fontsize=8, fontweight='bold')

# ── Chart 3 — Churn Rate by Segment ───────────────────────
ax3          = fig.add_subplot(gs[1, 2])
churn_by_seg = master.groupby('RFM_Segment')['churned'].mean() * 100
bars3 = ax3.barh(
    range(len(churn_by_seg)), churn_by_seg.values,
    color     = [colors_seg.get(s,'#999') for s in churn_by_seg.index],
    edgecolor = 'white', height=0.6
)
ax3.set_yticks(range(len(churn_by_seg)))
ax3.set_yticklabels(churn_by_seg.index, fontsize=8)
ax3.set_title('Churn Rate by Segment\n(Module 2 — Survival Analysis)', fontweight='bold', fontsize=10)
ax3.set_xlabel('Churn Rate (%)')
ax3.axvline(x=master['churned'].mean()*100, color='black', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f"Avg {master['churned'].mean()*100:.1f}%")
ax3.legend(fontsize=8)
ax3.set_facecolor('#FAFAFA')
ax3.grid(axis='x', alpha=0.3)
for bar, val in zip(bars3, churn_by_seg.values):
    ax3.text(val+0.5, bar.get_y()+bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')

# ── Chart 4 — Sentiment Distribution ──────────────────────
ax4 = fig.add_subplot(gs[1, 3])
sent_counts = pd.Series({'Positive': 494392, 'Neutral': 11582, 'Negative': 54803})
bars4 = ax4.bar(
    sent_counts.index, sent_counts.values/1000,
    color     = ['#1D9E75','#F39C12','#E74C3C'],
    edgecolor = 'white', linewidth=1.5, width=0.5
)
ax4.set_title('Review Sentiment Distribution\n(Module 3 — Sentiment Analysis)', fontweight='bold', fontsize=10)
ax4.set_ylabel('Reviews (thousands)')
ax4.set_facecolor('#FAFAFA')
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars4, sent_counts.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val/1000:.0f}K\n({val/sent_counts.sum()*100:.1f}%)',
             ha='center', fontsize=8, fontweight='bold')

# ── Chart 5 — Top Recommended Products ────────────────────
ax5 = fig.add_subplot(gs[2, :2])
top_recs = recs.groupby('Description')['Score'].sum().sort_values(ascending=False).head(10)
labels5  = [d[:35]+'...' if len(str(d))>35 else str(d) for d in top_recs.index]
ax5.barh(range(len(top_recs)), top_recs.values, color='#2196F3', edgecolor='white', height=0.7)
ax5.set_yticks(range(len(top_recs)))
ax5.set_yticklabels(labels5[::-1], fontsize=8)
ax5.invert_yaxis()
ax5.set_title('Top 10 Most Recommended Products\n(Module 1 — Recommendation System)',
              fontweight='bold', fontsize=10)
ax5.set_xlabel('Total Recommendation Score across all customers')
ax5.set_facecolor('#FAFAFA')
ax5.grid(axis='x', alpha=0.3)

# ── Chart 6 — RFM Score Distribution ──────────────────────
ax6 = fig.add_subplot(gs[2, 2:])
rfm['RFM_Score'].hist(bins=13, ax=ax6, color='#8E44AD', edgecolor='white', linewidth=1.5)
ax6.axvline(x=rfm['RFM_Score'].mean(), color='#E74C3C', linestyle='--', linewidth=2,
            label=f"Mean score = {rfm['RFM_Score'].mean():.1f}")
ax6.axvspan(13, 15.5, alpha=0.18, color='#1D9E75', label='Champions zone (score 13-15)')
ax6.set_title('RFM Score Distribution\n(Module 4 — Large Scale Mining)',
              fontweight='bold', fontsize=10)
ax6.set_xlabel('Combined RFM Score (3 = worst, 15 = best)')
ax6.set_ylabel('Number of Customers')
ax6.legend(fontsize=9)
ax6.set_facecolor('#FAFAFA')
ax6.grid(axis='y', alpha=0.3)

plt.savefig('/tmp/dashboard_page1.png', dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
plt.show()
print("Dashboard Page 1 — Executive Summary generated!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Dashboard Page 2: Operational Insights and Action Plan
# MAGIC
# MAGIC Four-panel operational dashboard with the survival analysis,
# MAGIC at-risk customer list, product sentiment analysis, and
# MAGIC prioritised business recommendations.

# COMMAND ----------

from lifelines import KaplanMeierFitter

fig2 = plt.figure(figsize=(20, 14))
fig2.patch.set_facecolor('#F8F9FA')
gs2  = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.32,
                         top=0.88, bottom=0.08, left=0.06, right=0.97)

fig2.text(0.5, 0.95, 'E-Commerce — Operational Insights and Business Action Plan',
          ha='center', fontsize=20, fontweight='bold', color='#1a1a2e')
fig2.text(0.5, 0.915,
          'Survival Analysis  |  Retention Targets  |  Product Quality Signals  |  Priority Recommendations',
          ha='center', fontsize=11, color='#555555')
fig2.add_artist(plt.Line2D([0.05, 0.95], [0.905, 0.905],
                color='#E74C3C', linewidth=2.5, transform=fig2.transFigure))

# ── Chart 1 — Survival Curves by Value Segment ────────────
ax1 = fig2.add_subplot(gs2[0, 0])
seg_colors2 = {
    'High Value'  : '#1D9E75',
    'Medium Value': '#F39C12',
    'Low Value'   : '#E74C3C'
}
for seg in ['High Value', 'Medium Value', 'Low Value']:
    mask = survival['segment'] == seg
    if mask.sum() < 10:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations      = survival.loc[mask, 'duration'].clip(lower=1),
        event_observed = survival.loc[mask, 'churned'],
        label          = f"{seg} (n={mask.sum():,})"
    )
    kmf.plot_survival_function(
        ax=ax1, ci_show=False,
        color=seg_colors2.get(seg,'#999'), linewidth=2.5
    )
ax1.set_title('Customer Survival by Value Segment\n(Module 2 — Survival Analysis)',
              fontweight='bold', fontsize=11)
ax1.set_xlabel('Days Since First Purchase')
ax1.set_ylabel('Survival Probability (% still active)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#FAFAFA')

# ── Chart 2 — At-Risk Customers ───────────────────────────
ax2            = fig2.add_subplot(gs2[0, 1])
at_risk_sorted = at_risk.nlargest(15, 'risk_score')
bar_colors_r   = ['#E74C3C' if s > 0.75 else '#F39C12' for s in at_risk_sorted['risk_score']]
ax2.barh(range(len(at_risk_sorted)), at_risk_sorted['risk_score'],
         color=bar_colors_r, edgecolor='white', height=0.7)
ax2.set_yticks(range(len(at_risk_sorted)))
ax2.set_yticklabels(
    [f"Customer {int(c)}" for c in at_risk_sorted['Customer_ID']],
    fontsize=8
)
ax2.set_title('Top 15 At-Risk Customers — Immediate Retention Targets\n(Module 2 — Survival Analysis)',
              fontweight='bold', fontsize=11)
ax2.set_xlabel('Risk Score (higher = more urgent action needed)')
ax2.axvline(x=0.75, color='#E74C3C', linestyle='--', linewidth=1.5, label='High risk threshold')
ax2.invert_yaxis()
ax2.set_facecolor('#FAFAFA')
ax2.grid(axis='x', alpha=0.3)
red_p    = mpatches.Patch(color='#E74C3C', label='Critical risk (> 0.75)')
orange_p = mpatches.Patch(color='#F39C12', label='High risk (0.60-0.75)')
ax2.legend(handles=[red_p, orange_p], fontsize=9, loc='lower right')

# ── Chart 3 — Product Sentiment Scatter ───────────────────
ax3       = fig2.add_subplot(gs2[1, 0])
prod_plot = prod_sent[prod_sent['review_count'] >= 10].copy()
scatter   = ax3.scatter(
    prod_plot['avg_sentiment'], prod_plot['review_count'],
    c=prod_plot['avg_sentiment'], cmap='RdYlGn',
    alpha=0.55, s=prod_plot['avg_rating'] * 12,
    vmin=-1, vmax=1
)
plt.colorbar(scatter, ax=ax3, label='VADER Sentiment Score')
ax3.axvline(x=0,   color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.axvline(x=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Positive threshold')
ax3.set_title('Product Sentiment vs Review Volume\n(Module 3 — Sentiment Analysis)',
              fontweight='bold', fontsize=11)
ax3.set_xlabel('Avg VADER Sentiment Score (-1.0 = negative, +1.0 = positive)')
ax3.set_ylabel('Number of Reviews')
ax3.legend(fontsize=9)
ax3.set_facecolor('#FAFAFA')
ax3.grid(True, alpha=0.3)

# ── Chart 4 — Business Recommendations ────────────────────
ax4 = fig2.add_subplot(gs2[1, 1])
ax4.set_facecolor('#FAFAFA')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_visible(False)
ax4.set_title('Priority Business Recommendations', fontweight='bold', fontsize=12, pad=12)

recommendations = [
    ('1', '#E74C3C', 'URGENT: Retention Campaign (Within 30 Days)',
     f"Target {len(at_risk):,} at-risk customers with\n15% discount. Est. recovery: £306,985"),
    ('2', '#F39C12', 'Champions VIP Programme',
     "39 Champions = 26.2% of revenue. Launch\npremium VIP programme to protect them"),
    ('3', '#1D9E75', 'Personalised Recommendations Deployment',
     "Deploy recommendation engine to 5,878\ncustomers. Mean confidence: 277.91"),
    ('4', '#2196F3', 'Product Quality Review',
     "5 products with sentiment < -0.5 need\nimmediate review or removal"),
    ('5', '#8E44AD', 'Lost Customer Re-engagement',
     "1,886 lost customers (32.1%). Launch\nwin-back campaign with personalised offers"),
]

y_pos = 0.86
for num, color, title, desc in recommendations:
    rect = FancyBboxPatch(
        (0.02, y_pos-0.14), 0.96, 0.13,
        boxstyle    = "round,pad=0.01",
        facecolor   = color + '18',
        edgecolor   = color,
        linewidth   = 1.8,
        transform   = ax4.transAxes
    )
    ax4.add_patch(rect)
    ax4.text(0.07, y_pos-0.04, f"{num}.", transform=ax4.transAxes,
             fontsize=13, fontweight='bold', color=color, va='top')
    ax4.text(0.13, y_pos-0.03, title, transform=ax4.transAxes,
             fontsize=9, fontweight='bold', color='#1a1a2e', va='top')
    ax4.text(0.13, y_pos-0.09, desc, transform=ax4.transAxes,
             fontsize=8, color='#444444', va='top', linespacing=1.5)
    y_pos -= 0.175

plt.savefig('/tmp/dashboard_page2.png', dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
plt.show()
print("Dashboard Page 2 — Operational Insights generated!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 8 — Save Dashboard Images to S3
# MAGIC
# MAGIC Uploading both dashboard pages to S3 for inclusion in the
# MAGIC technical report and business presentation.

# COMMAND ----------

import boto3

print("Uploading dashboard images to S3...")

s3_upload = boto3.client(
    's3',
    aws_access_key_id     = ACCESS_KEY,
    aws_secret_access_key = SECRET_KEY,
    region_name           = REGION
)

with open('/tmp/dashboard_page1.png', 'rb') as f:
    s3_upload.put_object(
        Bucket      = BUCKET,
        Key         = "outputs/dashboard_page1_executive_summary.png",
        Body        = f.read(),
        ContentType = 'image/png'
    )
print("  Uploaded: outputs/dashboard_page1_executive_summary.png")

with open('/tmp/dashboard_page2.png', 'rb') as f:
    s3_upload.put_object(
        Bucket      = BUCKET,
        Key         = "outputs/dashboard_page2_operational_insights.png",
        Body        = f.read(),
        ContentType = 'image/png'
    )
print("  Uploaded: outputs/dashboard_page2_operational_insights.png")

print()
print("Final file list in S3 outputs/:")
list_files("outputs/")

print()
print("=" * 65)
print("NOTEBOOK 06 — DASHBOARD SETUP COMPLETE")
print("=" * 65)
print()
print("Deliverables produced:")
print("  Delta Tables (6)    : ecom_project database in Unity Catalog")
print("  Dashboard Page 1    : outputs/dashboard_page1_executive_summary.png")
print("  Dashboard Page 2    : outputs/dashboard_page2_operational_insights.png")
print()
print("For interactive dashboard:")
print("  1. Go to Dashboards tab in left sidebar")
print("  2. Click 'Create dashboard'")
print("  3. Add visualizations using SQL queries from Section 4 above")
print("  4. Use table names: ecom_project.master_customers etc.")
print()
print("=" * 65)
print("PROJECT COMPLETE — ALL NOTEBOOKS FINISHED")
print("=" * 65)
print()
print("Remaining deliverables:")
print("  1. Technical Report (20-25 pages)")
print("  2. Business Presentation (10-12 slides)")
print("  3. README.md + requirements.txt")
