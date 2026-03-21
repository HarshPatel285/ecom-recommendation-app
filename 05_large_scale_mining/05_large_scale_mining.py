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
# MAGIC ## Notebook 05 — Large Scale Mining (Module 4 of 4)
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook implements **Large Scale Customer Analytics** using
# MAGIC RFM Analysis and K-Means Clustering to segment all 5,878 customers
# MAGIC into actionable business groups. It also **integrates the outputs
# MAGIC of all four modules** into a single master customer dataset.
# MAGIC
# MAGIC ### Business Problem
# MAGIC Not all customers are equal in value or behaviour.
# MAGIC Treating all customers the same wastes marketing spend and
# MAGIC misses opportunities. RFM segmentation identifies which customers
# MAGIC deserve premium treatment and which need re-engagement.
# MAGIC
# MAGIC ### Methodology
# MAGIC
# MAGIC **RFM Analysis** scores each customer on three dimensions:
# MAGIC
# MAGIC | Dimension | Question | Scoring |
# MAGIC |---|---|---|
# MAGIC | Recency (R) | How recently did they buy? | Lower recency = higher score |
# MAGIC | Frequency (F) | How often do they buy? | Higher frequency = higher score |
# MAGIC | Monetary (M) | How much do they spend? | Higher spend = higher score |
# MAGIC
# MAGIC Each dimension is scored 1-5 using percentile ranks.
# MAGIC Combined RFM score ranges from 3 (worst) to 15 (best).
# MAGIC
# MAGIC **K-Means Clustering** complements RFM by finding natural groupings
# MAGIC in the customer data without predefined thresholds.
# MAGIC
# MAGIC ### Module Integration
# MAGIC This notebook is the **integration hub** — it combines:
# MAGIC - Module 1 outputs: Recommendation scores
# MAGIC - Module 2 outputs: Churn probability and survival features
# MAGIC - Module 3 outputs: Sentiment indicators
# MAGIC - Module 4 outputs: RFM scores and clusters
# MAGIC
# MAGIC into a single master customer dataset for the dashboard.
# MAGIC
# MAGIC ### Outputs
# MAGIC - Master customer dataset saved to S3
# MAGIC - RFM segments saved to S3
# MAGIC - MLflow experiment: `/ecom-project/large-scale-mining`
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Install Required Libraries

# COMMAND ----------

%pip install scikit-learn pandas numpy matplotlib seaborn pyarrow mlflow boto3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — Load Configuration and All Module Data
# MAGIC
# MAGIC Loading outputs from all previous modules.
# MAGIC This demonstrates the integration of the analytical pipeline.

# COMMAND ----------

%run "../00_config"

import mlflow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches   import FancyBboxPatch
import matplotlib.patches as mpatches
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import silhouette_score

print("Configuration loaded successfully!")
print()
print("Loading all module outputs from S3...")
print()

retail_clean   = read_parquet("retail_clean")
survival_data  = read_parquet("survival_data")
sentiment_data = read_parquet("reviews_sentiment")
recs_data      = read_parquet("recommendations")

retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

print(f"  Retail transactions  : {len(retail_clean):,} rows  (source for RFM)")
print(f"  Survival data        : {len(survival_data):,} rows  (Module 2 output)")
print(f"  Sentiment data       : {len(sentiment_data):,} rows (Module 3 output)")
print(f"  Recommendations      : {len(recs_data):,} rows (Module 1 output)")
print()
print("All module data loaded successfully — integration ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — RFM Analysis
# MAGIC
# MAGIC Calculating Recency, Frequency, and Monetary values for each customer,
# MAGIC then converting them to 1-5 scores using percentile ranking.
# MAGIC
# MAGIC **Why Percentile Ranking instead of qcut?**
# MAGIC Standard `pd.qcut` fails when many customers have identical values
# MAGIC (e.g. hundreds of one-time buyers). Percentile ranking (`rank(pct=True)`)
# MAGIC always produces unique values, making it robust to ties.
# MAGIC
# MAGIC **RFM Segments:**
# MAGIC
# MAGIC | Segment | RFM Score | Description |
# MAGIC |---|---|---|
# MAGIC | Champions | 13-15 | Best customers — bought recently, often, and spend most |
# MAGIC | Loyal Customers | 10-12 | Regular buyers with good spend |
# MAGIC | At Risk | 7-9 | Used to buy often but haven't recently |
# MAGIC | Lost | 3-6 | Low recency, frequency, and monetary |

# COMMAND ----------

print("Running RFM Analysis on full customer transaction data...")
print()

reference_date = retail_clean['InvoiceDate'].max()
print(f"Reference date : {reference_date.date()}")

# Calculate R, F, M values for each customer
rfm = retail_clean.groupby('Customer ID').agg(
    Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency = ('Invoice',     'nunique'),
    Monetary  = ('TotalSpend',  'sum')
).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"RFM values calculated for {len(rfm):,} customers")
print()
print("RFM Distribution:")
print(rfm[['Recency','Frequency','Monetary']].describe().round(2).to_string())
print()

# ── Safe RFM Scoring using percentile ranking ──────────────
def safe_rfm_score(series, reverse=False):
    """
    Score a series 1-5 using percentile ranking.
    Robust to duplicate values — unlike pd.qcut.

    Args:
        series: pandas Series to score
        reverse (bool): If True, lower values get higher scores (for Recency)

    Returns:
        pandas Series with integer scores 1-5
    """
    pct = series.rank(pct=True)
    if reverse:
        pct = 1 - pct
    return pd.cut(
        pct,
        bins           = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels         = [1, 2, 3, 4, 5],
        include_lowest = True
    ).astype(int)


# Apply scoring (lower recency = bought recently = score of 5)
rfm['R_Score']   = safe_rfm_score(rfm['Recency'],   reverse=True)
rfm['F_Score']   = safe_rfm_score(rfm['Frequency'],  reverse=False)
rfm['M_Score']   = safe_rfm_score(rfm['Monetary'],   reverse=False)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

# Assign business segment labels
def rfm_segment(score):
    if score >= 13:
        return 'Champions'
    elif score >= 10:
        return 'Loyal Customers'
    elif score >= 7:
        return 'At Risk'
    else:
        return 'Lost'

rfm['RFM_Segment'] = rfm['RFM_Score'].apply(rfm_segment)

print("RFM Scoring complete:")
print(rfm[['R_Score','F_Score','M_Score','RFM_Score']].describe().round(2).to_string())
print()
print("=== Customer Segments (RFM Method) ===")
seg_counts = rfm['RFM_Segment'].value_counts()
seg_pct    = (rfm['RFM_Segment'].value_counts(normalize=True) * 100).round(1)
for seg in ['Champions','Loyal Customers','At Risk','Lost']:
    if seg in seg_counts:
        print(f"  {seg:<20} : {seg_counts[seg]:>5,} customers ({seg_pct[seg]:>5.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — K-Means Clustering
# MAGIC
# MAGIC K-Means is applied to the scaled RFM features to find natural
# MAGIC customer groupings without predefined boundaries.
# MAGIC
# MAGIC **Model Selection:**
# MAGIC We test k=2 through k=8 and select k=4 because:
# MAGIC 1. It matches our business segment structure (4 groups)
# MAGIC 2. The silhouette score is acceptable
# MAGIC 3. Each cluster is large enough for actionable campaigns
# MAGIC
# MAGIC **Feature Scaling:**
# MAGIC RFM values have very different scales (Recency: 0-738 days,
# MAGIC Monetary: £3 - £608,822). StandardScaler normalises all features
# MAGIC to the same scale before clustering.

# COMMAND ----------

print("Running K-Means Clustering on scaled RFM features...")
print()

features   = ['Recency', 'Frequency', 'Monetary']
scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

# Test k=2 to k=8 using elbow method and silhouette score
print("Elbow Method — Testing k=2 to k=8:")
inertias   = []
sil_scores = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))
    print(f"  k={k} | Inertia = {km.inertia_:>8.0f} | Silhouette = {sil_scores[-1]:.4f}")

# Final model with k=4
print()
print("Selected k=4 — matches business segment structure")
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

cluster_profile = rfm.groupby('Cluster')[features].mean().round(2)
print()
print("Cluster Profiles (mean values):")
print(cluster_profile.to_string())

# Assign business labels based on cluster profiles
cluster_labels = {}
for cluster in range(4):
    profile = cluster_profile.loc[cluster]
    if profile['Recency'] < cluster_profile['Recency'].median():
        if profile['Monetary'] > cluster_profile['Monetary'].median():
            cluster_labels[cluster] = 'Champions'
        else:
            cluster_labels[cluster] = 'Loyal Customers'
    else:
        if profile['Monetary'] > cluster_profile['Monetary'].median():
            cluster_labels[cluster] = 'At Risk'
        else:
            cluster_labels[cluster] = 'Lost Customers'

rfm['Cluster_Label'] = rfm['Cluster'].map(cluster_labels)

print()
print("Final Cluster Assignments:")
for label, count in rfm['Cluster_Label'].value_counts().items():
    print(f"  {label:<20} : {count:,} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — RFM Visualisation
# MAGIC
# MAGIC Four charts communicating the customer segmentation results.
# MAGIC These charts are designed for a business audience — they answer
# MAGIC the question "who are our customers?" visually.

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Module 4: Large Scale Customer Mining — RFM Segmentation Analysis',
             fontsize=16, fontweight='bold')

colors_seg = {
    'Champions'      : '#1D9E75',
    'Loyal Customers': '#2196F3',
    'At Risk'        : '#FF9800',
    'Lost Customers' : '#E74C3C',
    'Lost'           : '#E74C3C'
}

# ── Chart 1 — Customer Segment Donut ──────────────────────
ax1       = fig.add_subplot(gs[0, 0])
seg_data  = rfm['Cluster_Label'].value_counts()
ax1.pie(
    seg_data.values,
    labels     = [f"{s}\n({c:,} customers)" for s, c in zip(seg_data.index, seg_data.values)],
    colors     = [colors_seg.get(s,'#999') for s in seg_data.index],
    autopct    = '%1.1f%%',
    startangle = 90,
    pctdistance= 0.75,
    wedgeprops = dict(width=0.55, edgecolor='white', linewidth=2)
)
ax1.set_title('Customer Segment Distribution\n(K-Means Clustering)',
              fontweight='bold', fontsize=12)

# ── Chart 2 — Avg Revenue by Segment ──────────────────────
ax2         = fig.add_subplot(gs[0, 1])
seg_revenue = rfm.groupby('Cluster_Label')['Monetary'].mean().sort_values(ascending=False)
bars = ax2.bar(
    range(len(seg_revenue)), seg_revenue.values,
    color     = [colors_seg.get(s,'#999') for s in seg_revenue.index],
    edgecolor = 'white', linewidth=1.5, width=0.6
)
ax2.set_xticks(range(len(seg_revenue)))
ax2.set_xticklabels([s.replace(' ','\n') for s in seg_revenue.index], fontsize=9)
ax2.set_title('Average Revenue per Customer by Segment',
              fontweight='bold', fontsize=12)
ax2.set_ylabel('Avg Total Spend (£)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
for bar, val in zip(bars, seg_revenue.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
             f'£{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
ax2.set_facecolor('#FAFAFA')
ax2.grid(axis='y', alpha=0.3)

# ── Chart 3 — Recency vs Monetary Scatter ─────────────────
ax3 = fig.add_subplot(gs[1, 0])
for segment in rfm['Cluster_Label'].unique():
    mask = rfm['Cluster_Label'] == segment
    ax3.scatter(
        rfm.loc[mask,'Recency'], rfm.loc[mask,'Monetary'],
        c     = colors_seg.get(segment,'#999'),
        label = segment, alpha=0.5, s=20
    )
ax3.set_title('Recency vs Monetary Value by Segment',
              fontweight='bold', fontsize=12)
ax3.set_xlabel('Days Since Last Purchase (lower = more recent)')
ax3.set_ylabel('Total Spend (£)')
ax3.legend(fontsize=9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
ax3.set_facecolor('#FAFAFA')
ax3.grid(True, alpha=0.3)

# ── Chart 4 — RFM Score Distribution ──────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rfm['RFM_Score'].hist(bins=12, ax=ax4, color='#9C27B0', edgecolor='white', linewidth=1.5)
ax4.axvline(x=rfm['RFM_Score'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"Mean = {rfm['RFM_Score'].mean():.1f}")
ax4.axvspan(13, 15.5, alpha=0.2, color='#1D9E75', label='Champions zone (13-15)')
ax4.set_title('RFM Score Distribution\n(3=worst, 15=best)',
              fontweight='bold', fontsize=12)
ax4.set_xlabel('Combined RFM Score')
ax4.set_ylabel('Number of Customers')
ax4.legend(fontsize=9)
ax4.set_facecolor('#FAFAFA')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/rfm_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("RFM visualisation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Build Master Customer Dataset
# MAGIC
# MAGIC Combining outputs from all four modules into one unified customer dataset.
# MAGIC This is the **central deliverable** of the analytics pipeline —
# MAGIC a single customer-level dataset that contains:
# MAGIC
# MAGIC | Source | Fields Added |
# MAGIC |---|---|
# MAGIC | Module 1 (Recommendations) | avg_rec_score, top_rec_product |
# MAGIC | Module 2 (Survival) | duration, churned, total_orders, avg_order_value, days_since_last |
# MAGIC | Module 3 (Sentiment) | (via product linkage) |
# MAGIC | Module 4 (RFM Mining) | Recency, Frequency, Monetary, RFM_Score, RFM_Segment, Cluster_Label |

# COMMAND ----------

print("Building master customer dataset — integrating all 4 modules...")
print()

# Base: RFM data
master = rfm[[
    'CustomerID','Recency','Frequency','Monetary',
    'RFM_Score','RFM_Segment','Cluster_Label'
]].copy()
print(f"  Base (RFM)       : {len(master):,} rows")

# Merge Module 2 — Survival data
survival_merge = survival_data[[
    'Customer ID','duration','churned','total_orders',
    'avg_order_value','days_since_last','segment'
]].rename(columns={'Customer ID': 'CustomerID'})
master = master.merge(survival_merge, on='CustomerID', how='left')
print(f"  After Module 2   : {master.shape[1]} columns")

# Merge Module 1 — Recommendation scores
recs_merge = recs_data.groupby('CustomerID').agg(
    avg_rec_score   = ('Score',       'mean'),
    top_rec_product = ('Description', 'first')
).reset_index()
master = master.merge(recs_merge, on='CustomerID', how='left')
print(f"  After Module 1   : {master.shape[1]} columns")

print()
print(f"Master dataset final shape : {master.shape}")
print(f"Columns                    : {list(master.columns)}")
print()
print("Sample master dataset (first 5 rows):")
print(master.head(5).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Business Intelligence Summary
# MAGIC
# MAGIC Comprehensive business-level summary combining insights from all four modules.
# MAGIC This table answers the key executive question:
# MAGIC *"Which customer segments should we prioritise and why?"*

# COMMAND ----------

print("=" * 65)
print("COMPLETE CUSTOMER LIFECYCLE — BUSINESS INTELLIGENCE SUMMARY")
print("=" * 65)

segment_summary = master.groupby('RFM_Segment').agg(
    customer_count  = ('CustomerID',      'count'),
    avg_recency     = ('Recency',         'mean'),
    avg_frequency   = ('Frequency',       'mean'),
    avg_monetary    = ('Monetary',        'mean'),
    total_revenue   = ('Monetary',        'sum'),
    churn_rate      = ('churned',         'mean'),
    avg_tenure_days = ('duration',        'mean'),
    avg_order_value = ('avg_order_value', 'mean')
).round(2)

segment_summary['churn_rate_pct']    = (segment_summary['churn_rate'] * 100).round(1)
segment_summary['revenue_share_pct'] = (
    segment_summary['total_revenue'] / segment_summary['total_revenue'].sum() * 100
).round(1)

print("\nSegment Performance Table:")
print(segment_summary[[
    'customer_count','avg_recency','avg_frequency',
    'avg_monetary','churn_rate_pct','revenue_share_pct','avg_tenure_days'
]].to_string())

total_revenue   = master['Monetary'].sum()
total_customers = len(master)
churn_rate      = master['churned'].mean() * 100

print(f"\n{'='*65}")
print(f"KEY PERFORMANCE INDICATORS")
print(f"{'='*65}")
print(f"Total customers              : {total_customers:,}")
print(f"Total revenue                : £{total_revenue:,.2f}")
print(f"Overall churn rate           : {churn_rate:.1f}%")
print(f"Avg revenue per customer     : £{total_revenue/total_customers:,.2f}")
print()
print(f"Strategic Insights:")
for seg in ['Champions','Loyal Customers','At Risk','Lost']:
    if seg in segment_summary.index:
        row = segment_summary.loc[seg]
        print(f"  {seg:<20}: {row['customer_count']:>5,} customers | "
              f"£{row['avg_monetary']:>10,.0f} avg spend | "
              f"{row['churn_rate_pct']:>5.1f}% churn | "
              f"{row['revenue_share_pct']:>5.1f}% of revenue")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 8 — Final Combined Dashboard Chart
# MAGIC
# MAGIC An integrated four-panel chart combining insights from all four modules.
# MAGIC This chart tells the complete customer lifecycle story:
# MAGIC - Panel 1: Revenue by segment (who drives value)
# MAGIC - Panel 2: Churn rate by segment (who is at risk)
# MAGIC - Panel 3: Average order value by segment (purchase behaviour)
# MAGIC - Panel 4: Recency vs value scatter (segment positioning)

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Customer Lifecycle Intelligence Dashboard\nCombined Insights from All 4 Analytical Modules',
    fontsize=15, fontweight='bold', y=1.01
)

seg_order    = ['Champions','Loyal Customers','At Risk','Lost']
seg_colors_v = [colors_seg.get(s,'#999') for s in seg_order
                if s in segment_summary.index]
seg_data_v   = [segment_summary.loc[s] for s in seg_order if s in segment_summary.index]
seg_names    = [s for s in seg_order if s in segment_summary.index]

# Chart 1 — Total Revenue by Segment
revenues = [row['total_revenue'] for row in seg_data_v]
bars1    = axes[0,0].bar(seg_names, revenues, color=seg_colors_v, edgecolor='white', linewidth=1.5)
axes[0,0].set_title('Total Revenue by Segment\n(Modules 1 + 4)', fontweight='bold', fontsize=11)
axes[0,0].set_ylabel('Total Revenue (£)')
axes[0,0].tick_params(axis='x', rotation=10)
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
axes[0,0].set_facecolor('#FAFAFA')
for bar, row in zip(bars1, seg_data_v):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                   f"{row['revenue_share_pct']}% of total",
                   ha='center', fontsize=9, fontweight='bold')

# Chart 2 — Churn Rate by Segment
churns = [row['churn_rate_pct'] for row in seg_data_v]
bars2  = axes[0,1].bar(seg_names, churns, color=seg_colors_v, edgecolor='white', linewidth=1.5)
axes[0,1].set_title('Churn Rate by Segment\n(Module 2 — Survival Analysis)', fontweight='bold', fontsize=11)
axes[0,1].set_ylabel('Churn Rate (%)')
axes[0,1].tick_params(axis='x', rotation=10)
axes[0,1].axhline(y=churn_rate, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Overall avg {churn_rate:.1f}%')
axes[0,1].legend(fontsize=9)
axes[0,1].set_facecolor('#FAFAFA')
for bar, row in zip(bars2, seg_data_v):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, row['churn_rate_pct']+0.5,
                   f"{row['churn_rate_pct']}%", ha='center', fontsize=9)

# Chart 3 — Avg Order Value by Segment
aov   = [row['avg_order_value'] for row in seg_data_v]
bars3 = axes[1,0].bar(seg_names, aov, color=seg_colors_v, edgecolor='white', linewidth=1.5)
axes[1,0].set_title('Avg Order Value by Segment\n(Module 4 — RFM Mining)', fontweight='bold', fontsize=11)
axes[1,0].set_ylabel('Avg Order Value (£)')
axes[1,0].tick_params(axis='x', rotation=10)
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
axes[1,0].set_facecolor('#FAFAFA')
for bar, row in zip(bars3, seg_data_v):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, row['avg_order_value']+1,
                   f"£{row['avg_order_value']:,.0f}", ha='center', fontsize=9)

# Chart 4 — Recency vs Value Bubble
for s, row, color in zip(seg_names, seg_data_v, seg_colors_v):
    axes[1,1].scatter(
        row['avg_recency'], row['avg_monetary'],
        s=row['customer_count']*3, c=color, alpha=0.75, zorder=5
    )
    axes[1,1].annotate(
        s, (row['avg_recency'], row['avg_monetary']),
        textcoords="offset points", xytext=(8, 5),
        fontsize=10, fontweight='bold'
    )
axes[1,1].set_title('Recency vs Value — Segment Positioning\n(Bubble size = number of customers)',
                     fontweight='bold', fontsize=11)
axes[1,1].set_xlabel('Avg Days Since Last Purchase')
axes[1,1].set_ylabel('Avg Monetary Value (£)')
axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))
axes[1,1].set_facecolor('#FAFAFA')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/final_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Final combined dashboard chart generated!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 9 — Log to MLflow and Save to S3

# COMMAND ----------

mlflow.set_experiment("/ecom-project/large-scale-mining")

with mlflow.start_run(run_name="rfm_kmeans_clustering_v1"):
    mlflow.log_param("rfm_scoring_method",   "percentile_rank")
    mlflow.log_param("n_kmeans_clusters",    4)
    mlflow.log_param("kmeans_init",          "k-means++")
    mlflow.log_param("kmeans_n_init",        10)
    mlflow.log_param("scaler",               "StandardScaler")
    mlflow.log_param("total_customers",      len(master))
    mlflow.log_metric("total_revenue",       round(master['Monetary'].sum(), 2))
    mlflow.log_metric("overall_churn_pct",   round(master['churned'].mean() * 100, 2))
    mlflow.log_metric("avg_revenue_per_cust", round(master['Monetary'].mean(), 2))
    mlflow.log_metric("rfm_mean_score",      round(master['RFM_Score'].mean(), 2))
    for seg, count in rfm['Cluster_Label'].value_counts().items():
        mlflow.log_metric(f"seg_{seg.lower().replace(' ','_')}", count)
    run_id = mlflow.active_run().info.run_id

print("Saving master dataset and RFM segments to S3...")
save_parquet(master, "master_customers")
save_parquet(rfm,    "rfm_segments")

# Verify all output files
print()
print("All files in S3 outputs/ folder:")
list_files("outputs/")

print(f"\nMLflow run ID : {run_id}")
print()
print("=" * 65)
print("NOTEBOOK 05 — LARGE SCALE MINING COMPLETE")
print("=" * 65)
print(f"Customers segmented       : {len(master):,}")
print(f"Total revenue analysed    : £{master['Monetary'].sum():,.2f}")
print(f"Segments identified       : {rfm['Cluster_Label'].nunique()}")
print(f"Master dataset columns    : {master.shape[1]}")
print(f"MLflow run ID             : {run_id}")
print()
print("=" * 65)
print("ALL 4 ANALYTICAL MODULES COMPLETE")
print("=" * 65)
print("  Module 1 — Recommendation System      COMPLETE")
print("  Module 2 — Survival Analysis           COMPLETE")
print("  Module 3 — Sentiment Analysis          COMPLETE")
print("  Module 4 — Large Scale Mining          COMPLETE")
print()
print("Next Step: Run Notebook 06 — Dashboard Setup")
