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
# MAGIC ## Notebook 03 — Survival Analysis and Churn Modelling (Module 2 of 4)
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook applies **Survival Analysis** — a statistical method originally
# MAGIC developed in medical research — to model customer churn in an e-commerce context.
# MAGIC Survival analysis treats churn as an "event" and models the time until that event occurs.
# MAGIC
# MAGIC ### Business Problem
# MAGIC The retailer is losing 50.8% of its customers annually.
# MAGIC Understanding *when* customers churn and *what drives* churn
# MAGIC enables targeted intervention campaigns before customers leave.
# MAGIC
# MAGIC ### Methodology
# MAGIC Two complementary models are used:
# MAGIC
# MAGIC **1. Kaplan-Meier Estimator**
# MAGIC A non-parametric method that estimates the survival function —
# MAGIC the probability that a customer is still active after X days.
# MAGIC No assumptions are made about the shape of the survival curve.
# MAGIC
# MAGIC **2. Cox Proportional Hazards Model**
# MAGIC A semi-parametric regression model that identifies which customer
# MAGIC features (order frequency, spend, product diversity) most influence
# MAGIC the risk of churn. The hazard ratio tells us how much each feature
# MAGIC increases or decreases churn risk.
# MAGIC
# MAGIC ### Key Definitions
# MAGIC - **Duration**: Days from customer's first purchase to last purchase
# MAGIC - **Event (Churned)**: 1 if the customer made no purchase in the last 90 days
# MAGIC - **Censored**: Customers who have not yet churned (still active)
# MAGIC
# MAGIC ### Outputs
# MAGIC - Customer survival data saved to S3
# MAGIC - 497 at-risk customers identified for retention campaign
# MAGIC - MLflow experiment: `/ecom-project/survival-analysis`
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Install Required Libraries

# COMMAND ----------

%pip install lifelines pandas numpy matplotlib seaborn pyarrow mlflow boto3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — Load Configuration and Cleaned Data

# COMMAND ----------

%run "../00_config"

import mlflow
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

print("Configuration loaded successfully!")
print()
print("Loading cleaned retail data from S3...")
retail_clean = read_parquet("retail_clean")
retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

print(f"  Rows loaded  : {len(retail_clean):,}")
print(f"  Date range   : {retail_clean['InvoiceDate'].min().date()} → {retail_clean['InvoiceDate'].max().date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — Engineer Survival Features
# MAGIC
# MAGIC Creating customer-level features required for survival analysis.
# MAGIC
# MAGIC **Churn Definition:**
# MAGIC A customer is defined as "churned" if they made no purchase
# MAGIC in the 90 days before the last transaction date in the dataset (2011-12-09).
# MAGIC 90 days is a standard retail industry churn threshold.
# MAGIC
# MAGIC **Feature Engineering:**
# MAGIC
# MAGIC | Feature | Calculation | Use |
# MAGIC |---|---|---|
# MAGIC | duration | last_purchase - first_purchase (days) | Survival time |
# MAGIC | churned | days_since_last > 90 → 1, else 0 | Event indicator |
# MAGIC | purchase_frequency | total_orders / tenure_months | Cox covariate |
# MAGIC | avg_order_value | total_spend / total_orders | Cox covariate |
# MAGIC | unique_products | count of distinct StockCodes | Cox covariate |

# COMMAND ----------

print("Engineering customer-level survival features...")
print()

# Reference date = last date in dataset
reference_date = retail_clean['InvoiceDate'].max()
print(f"Reference date (last transaction) : {reference_date.date()}")
print(f"Churn definition                  : No purchase in last 90 days")
print()

# Aggregate transaction data to customer level
customer_stats = retail_clean.groupby('Customer ID').agg(
    first_purchase    = ('InvoiceDate', 'min'),
    last_purchase     = ('InvoiceDate', 'max'),
    total_orders      = ('Invoice',     'nunique'),
    total_spend       = ('TotalSpend',  'sum'),
    total_items       = ('Quantity',    'sum'),
    unique_products   = ('StockCode',   'nunique'),
    avg_order_value   = ('TotalSpend',  'mean'),
    favourite_country = ('Country',     lambda x: x.mode()[0])
).reset_index()

# Calculate survival columns
customer_stats['duration']       = (customer_stats['last_purchase']  - customer_stats['first_purchase']).dt.days
customer_stats['days_since_last'] = (reference_date - customer_stats['last_purchase']).dt.days
customer_stats['churned']        = (customer_stats['days_since_last'] > 90).astype(int)
customer_stats['tenure_months']  = (customer_stats['duration'] / 30).clip(lower=0.1)
customer_stats['purchase_frequency'] = (customer_stats['total_orders'] / customer_stats['tenure_months'])
customer_stats['duration']       = customer_stats['duration'].clip(lower=1)

# Segment by value for stratified analysis
spend_median = customer_stats['total_spend'].median()
spend_75th   = customer_stats['total_spend'].quantile(0.75)

def segment_customer(spend):
    if spend >= spend_75th:
        return 'High Value'
    elif spend >= spend_median:
        return 'Medium Value'
    else:
        return 'Low Value'

customer_stats['segment'] = customer_stats['total_spend'].apply(segment_customer)

print(f"Customer survival dataset created:")
print(f"  Total customers     : {len(customer_stats):,}")
print(f"  Churned customers   : {customer_stats['churned'].sum():,} ({customer_stats['churned'].mean()*100:.1f}%)")
print(f"  Active customers    : {(customer_stats['churned']==0).sum():,} ({(customer_stats['churned']==0).mean()*100:.1f}%)")
print(f"  Avg tenure          : {customer_stats['duration'].mean():.0f} days")
print(f"  Avg total spend     : £{customer_stats['total_spend'].mean():,.2f}")
print()
print("Sample survival data (first 8 customers):")
print(customer_stats[['Customer ID','duration','churned','total_orders',
                       'total_spend','days_since_last','segment']].head(8).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — Kaplan-Meier Survival Curve (Overall)
# MAGIC
# MAGIC The Kaplan-Meier curve answers:
# MAGIC *"What percentage of our customers are still active after X days?"*
# MAGIC
# MAGIC The shaded area represents the 95% confidence interval.
# MAGIC Vertical dashed lines mark key business milestones (90, 180, 365 days).
# MAGIC The survival probability at each milestone is shown in the annotation box.

# COMMAND ----------

print("Fitting Kaplan-Meier survival model...")

kmf = KaplanMeierFitter()
kmf.fit(
    durations      = customer_stats['duration'],
    event_observed = customer_stats['churned'],
    label          = 'All Customers'
)

# Calculate survival at key milestones
prob_90  = float(kmf.survival_function_at_times([90]).values[0])
prob_180 = float(kmf.survival_function_at_times([180]).values[0])
prob_365 = float(kmf.survival_function_at_times([365]).values[0])

fig, ax = plt.subplots(figsize=(12, 6))

kmf.plot_survival_function(ax=ax, ci_show=True, color='#2196F3', linewidth=2.5)

# Reference lines at business milestones
ax.axvline(x=90,  color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='90 days')
ax.axvline(x=180, color='red',    linestyle='--', linewidth=1.5, alpha=0.8, label='180 days')
ax.axvline(x=365, color='purple', linestyle='--', linewidth=1.5, alpha=0.8, label='365 days')

ax.set_title(
    'Customer Survival Curve — Module 2: Survival Analysis\nWhat % of customers remain active over time?',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Days since first purchase', fontsize=11)
ax.set_ylabel('Survival Probability (% still active)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Annotation box with key statistics
textstr = (
    f'Survival at 90 days  : {prob_90*100:.1f}%\n'
    f'Survival at 180 days : {prob_180*100:.1f}%\n'
    f'Survival at 365 days : {prob_365*100:.1f}%'
)
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('/tmp/survival_curve_overall.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Kaplan-Meier Results:")
print(f"  Survival at 90 days  : {prob_90*100:.1f}%  ({100-prob_90*100:.1f}% have churned by 90 days)")
print(f"  Survival at 180 days : {prob_180*100:.1f}%  ({100-prob_180*100:.1f}% have churned by 180 days)")
print(f"  Survival at 365 days : {prob_365*100:.1f}%  ({100-prob_365*100:.1f}% have churned by one year)")
print(f"  Median survival time : {kmf.median_survival_time_:.0f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — Survival by Customer Segment
# MAGIC
# MAGIC Stratifying the survival analysis by customer value segment reveals
# MAGIC dramatically different churn patterns across segments.
# MAGIC
# MAGIC **Business Implication:**
# MAGIC High-value customers should receive premium retention programmes
# MAGIC because they have much longer lifetimes and higher revenue contribution.
# MAGIC Low-value customers need early intervention — they churn very quickly.

# COMMAND ----------

print("Fitting segment-stratified Kaplan-Meier models...")
print()

fig, ax = plt.subplots(figsize=(12, 6))

colors_seg   = {'High Value': '#4CAF50', 'Medium Value': '#FF9800', 'Low Value': '#F44336'}
median_times = {}

for segment in ['High Value', 'Medium Value', 'Low Value']:
    mask = customer_stats['segment'] == segment
    kmf_seg = KaplanMeierFitter()
    kmf_seg.fit(
        durations      = customer_stats.loc[mask, 'duration'],
        event_observed = customer_stats.loc[mask, 'churned'],
        label          = f'{segment} (n={mask.sum():,})'
    )
    kmf_seg.plot_survival_function(
        ax=ax, ci_show=False, color=colors_seg[segment], linewidth=2.5
    )
    median_times[segment] = kmf_seg.median_survival_time_

ax.set_title(
    'Customer Survival by Value Segment — Module 2: Survival Analysis\nHigh-value customers stay significantly longer',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Days since first purchase', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('/tmp/survival_by_segment.png', dpi=150, bbox_inches='tight')
plt.show()

print("Median survival time by customer value segment:")
for seg, time in median_times.items():
    churn_seg = customer_stats[customer_stats['segment'] == seg]['churned'].mean() * 100
    print(f"  {seg:<15} : {time:.0f} days median — {churn_seg:.1f}% churn rate")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Cox Proportional Hazards Model
# MAGIC
# MAGIC The Cox model identifies **which customer features drive churn risk**.
# MAGIC
# MAGIC **Interpreting Hazard Ratios:**
# MAGIC - **Negative coefficient** = feature REDUCES churn risk (protective factor)
# MAGIC - **Positive coefficient** = feature INCREASES churn risk (risk factor)
# MAGIC
# MAGIC All continuous features are log-transformed to handle skewed distributions
# MAGIC and improve model stability. A penalizer of 0.1 is applied for regularisation.

# COMMAND ----------

print("Fitting Cox Proportional Hazards model...")
print()

# Prepare Cox model features
cox_features = [
    'duration', 'churned', 'total_orders', 'total_spend',
    'avg_order_value', 'unique_products', 'purchase_frequency'
]
cox_df = customer_stats[cox_features].copy()

# Log-transform skewed continuous features
for col in ['total_orders','total_spend','avg_order_value',
            'unique_products','purchase_frequency']:
    cox_df[col] = np.log1p(cox_df[col])

cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Customers in Cox model : {len(cox_df):,}")
print()

# Fit Cox model with regularisation
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col='duration', event_col='churned')

print("Cox Model Summary:")
cph.print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Cox Model Visualisation
# MAGIC
# MAGIC **Chart 1 — Log Hazard Ratios**
# MAGIC Red bars indicate features that increase churn risk.
# MAGIC Green bars indicate features that reduce churn risk.
# MAGIC
# MAGIC **Chart 2 — Survival by Purchase Frequency**
# MAGIC Shows how purchase frequency affects survival probability.
# MAGIC Customers who buy more frequently (frequency=3.0) have significantly
# MAGIC higher survival probability than infrequent buyers (frequency=0.5).

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Module 2: Cox Model — What Drives Customer Churn?', fontsize=14, fontweight='bold')

# ── Chart 1 — Hazard Ratios ────────────────────────────────
coef_df = pd.DataFrame({
    'feature'  : cph.params_.index,
    'coef'     : cph.params_.values,
    'exp_coef' : np.exp(cph.params_.values)
}).sort_values('coef', ascending=True)

colors_bar = ['#F44336' if c > 0 else '#4CAF50' for c in coef_df['coef']]
axes[0].barh(coef_df['feature'], coef_df['coef'], color=colors_bar, edgecolor='white')
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].set_title('Log Hazard Ratios\n(Red = increases churn risk, Green = reduces it)',
                   fontweight='bold', fontsize=11)
axes[0].set_xlabel('Log Hazard Ratio')
axes[0].set_facecolor('#FAFAFA')
axes[0].grid(axis='x', alpha=0.3)

# ── Chart 2 — Partial Effects on Outcome ──────────────────
cph.plot_partial_effects_on_outcome(
    covariates='purchase_frequency', values=[0.5, 1.0, 2.0, 3.0],
    ax=axes[1], cmap='RdYlGn'
)
axes[1].set_title('Survival by Purchase Frequency\n(Higher frequency = lower churn risk)',
                   fontweight='bold', fontsize=11)
axes[1].set_xlabel('Days', fontsize=10)
axes[1].set_ylabel('Survival Probability', fontsize=10)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
axes[1].set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('/tmp/cox_model_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("Cox Model — Key Findings:")
for _, row in coef_df.sort_values('coef').iterrows():
    direction = "REDUCES" if row['coef'] < 0 else "INCREASES"
    print(f"  {row['feature']:<25} : {direction} churn risk (coefficient = {row['coef']:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 8 — Identify At-Risk Customers
# MAGIC
# MAGIC Identifying customers who are currently active but showing signs of
# MAGIC impending churn — defined as customers who have not purchased in 60-90 days.
# MAGIC
# MAGIC **Risk Score Calculation:**
# MAGIC The risk score combines three factors:
# MAGIC - Recency (how recently they last purchased) — 50% weight
# MAGIC - Purchase frequency (how often they buy) — 30% weight
# MAGIC - Total spend (higher value = higher priority) — 20% weight
# MAGIC
# MAGIC **Business Value:**
# MAGIC These 497 customers represent £1,023,283 in historical spend.
# MAGIC Industry research suggests a 30% win-back rate with targeted campaigns,
# MAGIC meaning a retention campaign could recover approximately £306,985.

# COMMAND ----------

print("Identifying at-risk customers for retention campaign...")
print()

# At-risk = active customers who have not purchased in 60-90 days
active_customers = customer_stats[customer_stats['churned'] == 0].copy()
at_risk = active_customers[active_customers['days_since_last'].between(60, 90)].copy()

# Calculate risk score
at_risk['risk_score'] = (
    at_risk['days_since_last'] / 90 * 0.5 +
    (1 / at_risk['total_orders'].clip(lower=1)) * 0.3 +
    (1 / at_risk['total_spend'].clip(lower=1))  * 0.2
).round(4)

at_risk = at_risk.sort_values('risk_score', ascending=False)
avg_spend = at_risk['total_spend'].mean()

print(f"Retention Campaign Target Analysis:")
print(f"  Total active customers          : {len(active_customers):,}")
print(f"  At-risk customers (60-90 days)  : {len(at_risk):,}")
print(f"  Avg historical spend per customer: £{avg_spend:,.2f}")
print()
print(f"Business Case for Retention Campaign:")
print(f"  At-risk customers               : {len(at_risk):,}")
print(f"  Total revenue at risk           : £{len(at_risk) * avg_spend:,.2f}")
print(f"  If 30% retained (industry avg)  : £{len(at_risk) * avg_spend * 0.30:,.2f} recovered")
print()
print("Top 10 highest-risk customers (immediate action required):")
print(at_risk[['Customer ID','days_since_last','total_orders',
               'total_spend','risk_score']].head(10).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 9 — Log to MLflow and Save to S3

# COMMAND ----------

mlflow.set_experiment("/ecom-project/survival-analysis")

with mlflow.start_run(run_name="survival_analysis_v1"):
    mlflow.log_param("churn_definition_days",  90)
    mlflow.log_param("model_type",             "kaplan_meier + cox_proportional_hazards")
    mlflow.log_param("total_customers",        len(customer_stats))
    mlflow.log_param("cox_penalizer",          0.1)
    mlflow.log_param("at_risk_window_days",    "60-90")
    mlflow.log_metric("churn_rate_pct",        round(customer_stats['churned'].mean() * 100, 2))
    mlflow.log_metric("median_survival_days",  float(kmf.median_survival_time_))
    mlflow.log_metric("survival_at_90_days",   round(prob_90 * 100, 2))
    mlflow.log_metric("survival_at_180_days",  round(prob_180 * 100, 2))
    mlflow.log_metric("survival_at_365_days",  round(prob_365 * 100, 2))
    mlflow.log_metric("at_risk_customers",     len(at_risk))
    mlflow.log_metric("revenue_at_risk",       round(len(at_risk) * avg_spend, 2))
    mlflow.log_metric("potential_recovery",    round(len(at_risk) * avg_spend * 0.30, 2))
    run_id = mlflow.active_run().info.run_id

print("Saving survival analysis results to S3...")
save_parquet(customer_stats, "survival_data")
save_parquet(at_risk,        "at_risk")

print(f"\nMLflow run ID : {run_id}")
print()
print("=" * 60)
print("NOTEBOOK 03 — SURVIVAL ANALYSIS COMPLETE")
print("=" * 60)
print(f"Churn rate            : {customer_stats['churned'].mean()*100:.1f}%")
print(f"Median survival       : {kmf.median_survival_time_:.0f} days")
print(f"Survival at 1 year    : {prob_365*100:.1f}%")
print(f"At-risk customers     : {len(at_risk):,}")
print(f"Revenue at risk       : £{len(at_risk) * avg_spend:,.2f}")
print(f"Potential recovery    : £{len(at_risk) * avg_spend * 0.30:,.2f}")
print(f"MLflow run ID         : {run_id}")
print()
print("Next Step: Run Notebook 04 — Sentiment Analysis")
