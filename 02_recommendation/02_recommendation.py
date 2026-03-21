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
# MAGIC ## Notebook 02 — Recommendation System (Module 1 of 4)
# MAGIC
# MAGIC ### Overview
# MAGIC This notebook implements a **Collaborative Filtering Recommendation System**
# MAGIC using cosine similarity to generate personalised product recommendations
# MAGIC for all 5,878 customers in the dataset.
# MAGIC
# MAGIC ### Business Problem
# MAGIC The e-commerce retailer currently shows the same products to all customers.
# MAGIC Personalised recommendations increase average order value and reduce churn
# MAGIC by showing customers products relevant to their purchase history.
# MAGIC Industry research (McKinsey, 2023) shows personalised recommendations
# MAGIC drive 35% higher average order value.
# MAGIC
# MAGIC ### Methodology
# MAGIC Collaborative Filtering is based on the principle that customers who bought
# MAGIC similar products in the past will continue to have similar preferences.
# MAGIC The algorithm:
# MAGIC 1. Builds a customer × product purchase matrix
# MAGIC 2. Calculates cosine similarity between all customer pairs
# MAGIC 3. For each customer, finds the 10 most similar customers
# MAGIC 4. Recommends products those similar customers bought (but the target has not)
# MAGIC
# MAGIC ### Outputs
# MAGIC - 29,390 personalised recommendations (top 5 per customer)
# MAGIC - Saved to `s3://ecom-analytics-project-2026/outputs/all_customer_recommendations.parquet`
# MAGIC - MLflow experiment: `/ecom-project/recommendation-model`
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 — Install Required Libraries

# COMMAND ----------

%pip install scikit-learn pandas numpy matplotlib pyarrow mlflow boto3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 — Load Configuration and Cleaned Data
# MAGIC
# MAGIC Loading the central configuration and the cleaned retail dataset
# MAGIC produced by Notebook 01.

# COMMAND ----------

%run "../00_config"

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse              import csr_matrix

print("Configuration loaded successfully!")
print()
print("Loading cleaned retail data from S3...")
retail_clean = read_parquet("retail_clean")
print(f"  Retail clean : {len(retail_clean):,} rows loaded")
print(f"  Columns      : {list(retail_clean.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3 — Build Customer-Product Purchase Matrix
# MAGIC
# MAGIC The foundation of collaborative filtering is a **customer-product interaction matrix**
# MAGIC where each row represents a customer and each column represents a product.
# MAGIC The cell value is the total quantity the customer purchased of that product.
# MAGIC
# MAGIC This matrix will be sparse (mostly zeros) because most customers
# MAGIC buy only a small subset of the product catalogue.
# MAGIC A density of 1-5% is normal and expected for e-commerce datasets.

# COMMAND ----------

print("Building customer-product interaction matrix...")
print()

# Aggregate: total quantity each customer bought of each product
purchase_matrix = retail_clean.groupby(
    ['Customer ID', 'StockCode']
)['Quantity'].sum().reset_index()
purchase_matrix.columns = ['CustomerID', 'StockCode', 'PurchaseCount']
purchase_matrix          = purchase_matrix[purchase_matrix['PurchaseCount'] > 0]

print(f"Interaction matrix statistics:")
print(f"  Total interactions  : {len(purchase_matrix):,}")
print(f"  Unique customers    : {purchase_matrix['CustomerID'].nunique():,}")
print(f"  Unique products     : {purchase_matrix['StockCode'].nunique():,}")
print(f"  Avg interactions    : {len(purchase_matrix)/purchase_matrix['CustomerID'].nunique():.1f} per customer")

# Create pivot table: rows = customers, columns = products, values = quantity
customer_product = purchase_matrix.pivot_table(
    index      = 'CustomerID',
    columns    = 'StockCode',
    values     = 'PurchaseCount',
    fill_value = 0
)

density = (customer_product > 0).sum().sum() / (customer_product.shape[0] * customer_product.shape[1]) * 100

print()
print(f"Pivot table created:")
print(f"  Matrix shape        : {customer_product.shape[0]:,} customers × {customer_product.shape[1]:,} products")
print(f"  Matrix density      : {density:.2f}% (non-zero cells)")
print(f"  Matrix sparsity     : {100-density:.2f}% (zero cells — expected for e-commerce)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 4 — Train Collaborative Filtering Model
# MAGIC
# MAGIC **Cosine Similarity** measures the angle between two customer purchase vectors.
# MAGIC A score of 1.0 means customers bought identical products in identical quantities.
# MAGIC A score of 0.0 means customers share no common purchases.
# MAGIC
# MAGIC We use **Sparse Matrix Representation** (`csr_matrix`) for efficiency —
# MAGIC storing only non-zero values reduces memory usage significantly for a
# MAGIC sparse matrix like our customer-product matrix.

# COMMAND ----------

print("Training collaborative filtering model using cosine similarity...")
print("This may take 1-2 minutes for a 5,878 × 5,878 similarity matrix...")
print()

# Convert to sparse matrix for memory efficiency
matrix_values       = csr_matrix(customer_product.values)

# Calculate cosine similarity between all customer pairs
# Result: 5,878 × 5,878 matrix where [i,j] = similarity of customers i and j
customer_similarity = cosine_similarity(matrix_values)

# Convert to DataFrame for easy lookup
customer_sim_df = pd.DataFrame(
    customer_similarity,
    index   = customer_product.index,
    columns = customer_product.index
)

print(f"Similarity matrix computed:")
print(f"  Shape               : {customer_sim_df.shape}")
print(f"  Diagonal values     : 1.0 (each customer is identical to themselves)")
print(f"  Off-diagonal range  : {customer_sim_df.values[customer_sim_df.values < 1.0].min():.4f} → {customer_sim_df.values[customer_sim_df.values < 1.0].max():.4f}")
print()
print("Sample similarity scores (first 5 customers):")
print(customer_sim_df.iloc[:5, :5].round(3).to_string())
print()
print("Model training COMPLETE!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 5 — Recommendation Function
# MAGIC
# MAGIC The recommendation function follows these steps for each customer:
# MAGIC
# MAGIC 1. **Find similar customers**: Sort all other customers by similarity score, take top 10
# MAGIC 2. **Collect candidate products**: Get all products those similar customers purchased
# MAGIC 3. **Calculate weighted scores**: Multiply purchase quantities by similarity score
# MAGIC 4. **Remove already-bought products**: Exclude products the customer has already purchased
# MAGIC 5. **Rank and return top N**: Sort remaining products by weighted score, return top N

# COMMAND ----------

def get_recommendations(customer_id, n_recommendations=10):
    """
    Generate personalised product recommendations for a customer.

    Uses collaborative filtering — finds customers with similar purchase
    histories and recommends products they bought that the target customer has not.

    Args:
        customer_id: Customer ID to generate recommendations for
        n_recommendations (int): Number of recommendations to return

    Returns:
        pandas.DataFrame with columns: StockCode, Score, Description
    """
    if customer_id not in customer_sim_df.index:
        return f"Customer {customer_id} not found in dataset"

    # Step 1 — Find top 10 most similar customers (exclude self at index 0)
    similar_customers    = customer_sim_df[customer_id] \
                           .sort_values(ascending=False)[1:11]
    similar_customer_ids = similar_customers.index.tolist()
    similar_purchases    = customer_product.loc[similar_customer_ids]

    # Step 2 — Calculate weighted purchase scores
    weighted_scores = pd.Series(0.0, index=customer_product.columns)
    for sim_customer, sim_score in similar_customers.items():
        weighted_scores += similar_purchases.loc[sim_customer] * sim_score

    # Step 3 — Remove products already purchased by target customer
    already_bought  = customer_product.loc[customer_id]
    already_bought  = already_bought[already_bought > 0].index.tolist()
    weighted_scores = weighted_scores.drop(
        labels=[p for p in already_bought if p in weighted_scores.index]
    )

    # Step 4 — Get top N recommendations with product descriptions
    top_recs     = weighted_scores.sort_values(ascending=False).head(n_recommendations)
    product_desc = retail_clean[['StockCode','Description']] \
                   .drop_duplicates('StockCode') \
                   .set_index('StockCode')['Description']

    result = pd.DataFrame({
        'StockCode'   : top_recs.index,
        'Score'       : top_recs.values.round(3),
        'Description' : top_recs.index.map(lambda x: product_desc.get(x, 'Unknown'))
    }).reset_index(drop=True)
    result.index = result.index + 1  # Start ranking from 1
    return result


print("Recommendation function built successfully!")
print()
print("Testing with sample customer...")
sample_customer = customer_product.index[0]
print(f"Top 10 recommendations for Customer {sample_customer}:")
print("=" * 65)
recs_sample = get_recommendations(sample_customer, n_recommendations=10)
print(recs_sample.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 6 — Model Evaluation
# MAGIC
# MAGIC Evaluating the recommendation model using three metrics:
# MAGIC
# MAGIC | Metric | Description | Ideal Value |
# MAGIC |---|---|---|
# MAGIC | Catalog Coverage | % of products that can be recommended | Higher is better |
# MAGIC | Avg Customer Similarity | How similar the matched customers are | Higher = better matches |
# MAGIC | Avg Recommendation Score | Weighted confidence of recommendations | Higher is better |

# COMMAND ----------

print("Evaluating recommendation model quality...")
print()

# Sample 100 customers for evaluation
sample_customers = customer_product.index[:100].tolist()

# Metric 1 — Catalog Coverage
recommended_products = set()
for cust in sample_customers:
    recs = get_recommendations(cust, n_recommendations=10)
    if isinstance(recs, pd.DataFrame):
        recommended_products.update(recs['StockCode'].tolist())
coverage = len(recommended_products) / len(customer_product.columns) * 100

# Metric 2 — Average Customer Similarity
avg_similarity = []
for cust in sample_customers[:50]:
    top_sim = customer_sim_df[cust].sort_values(ascending=False)[1:6]
    avg_similarity.append(top_sim.mean())
mean_similarity = np.mean(avg_similarity)

# Metric 3 — Average Recommendation Score
avg_rec_score = []
for cust in sample_customers[:50]:
    recs = get_recommendations(cust, n_recommendations=10)
    if isinstance(recs, pd.DataFrame) and len(recs) > 0:
        avg_rec_score.append(recs['Score'].mean())
mean_rec_score = np.mean(avg_rec_score)

print("=== Recommendation Model Evaluation ===")
print()
print(f"Dataset Statistics:")
print(f"  Total customers           : {customer_product.shape[0]:,}")
print(f"  Total products            : {customer_product.shape[1]:,}")
print(f"  Total interactions        : {len(purchase_matrix):,}")
print()
print(f"Model Performance Metrics:")
print(f"  Catalog coverage          : {coverage:.1f}%")
print(f"  Avg customer similarity   : {mean_similarity:.4f}")
print(f"  Avg recommendation score  : {mean_rec_score:.4f}")
print()
print(f"Business Interpretation:")
print(f"  {coverage:.1f}% of the {customer_product.shape[1]:,} product catalogue can be recommended")
print(f"  Average customer similarity of {mean_similarity:.4f} indicates meaningful purchase patterns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 7 — Log Experiment to MLflow
# MAGIC
# MAGIC All model parameters and metrics are logged to **MLflow** for experiment tracking.
# MAGIC MLflow is built into Databricks and provides:
# MAGIC - Complete audit trail of all model runs
# MAGIC - Parameter and metric comparison across runs
# MAGIC - Model versioning and registration
# MAGIC - Evidence of cloud-native model lifecycle management

# COMMAND ----------

mlflow.set_experiment("/ecom-project/recommendation-model")

with mlflow.start_run(run_name="collaborative_filtering_v1"):

    # Log model parameters
    mlflow.log_param("model_type",            "collaborative_filtering")
    mlflow.log_param("similarity_metric",     "cosine_similarity")
    mlflow.log_param("n_recommendations",     10)
    mlflow.log_param("n_similar_customers",   10)
    mlflow.log_param("matrix_shape_customers", customer_product.shape[0])
    mlflow.log_param("matrix_shape_products",  customer_product.shape[1])
    mlflow.log_param("matrix_density_pct",    round(density, 2))

    # Log evaluation metrics
    mlflow.log_metric("catalog_coverage_pct",      round(coverage, 2))
    mlflow.log_metric("avg_customer_similarity",   round(mean_similarity, 4))
    mlflow.log_metric("avg_recommendation_score",  round(mean_rec_score, 4))
    mlflow.log_metric("total_interactions",        len(purchase_matrix))

    run_id = mlflow.active_run().info.run_id

print(f"MLflow experiment logged successfully!")
print(f"  Experiment : /ecom-project/recommendation-model")
print(f"  Run name   : collaborative_filtering_v1")
print(f"  Run ID     : {run_id}")
print()
print("View this run in the Experiments tab in the left sidebar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 8 — Generate Recommendations for All Customers
# MAGIC
# MAGIC Generating top-5 personalised product recommendations for all 5,878 customers.
# MAGIC Progress is printed every 500 customers to monitor completion.

# COMMAND ----------

print(f"Generating recommendations for all {customer_product.shape[0]:,} customers...")
print(f"Generating top 5 recommendations per customer...")
print()

all_recommendations = []

for i, customer_id in enumerate(customer_product.index):
    recs = get_recommendations(customer_id, n_recommendations=5)
    if isinstance(recs, pd.DataFrame):
        recs['CustomerID'] = customer_id
        all_recommendations.append(recs)
    if (i + 1) % 500 == 0:
        print(f"  Progress: {i+1:,} / {customer_product.shape[0]:,} customers processed...")

all_recs_df = pd.concat(all_recommendations, ignore_index=True)

print()
print(f"Recommendations generated:")
print(f"  Total customers     : {customer_product.shape[0]:,}")
print(f"  Total recs rows     : {len(all_recs_df):,}")
print(f"  Avg recs/customer   : {len(all_recs_df)/customer_product.shape[0]:.1f}")
print()
print("Sample recommendations (first 10 rows):")
print(all_recs_df.head(10).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 9 — Business Insights Visualisation
# MAGIC
# MAGIC Two charts to communicate the recommendation system results to a business audience.
# MAGIC
# MAGIC **Chart 1 — Top 15 Most Recommended Products**
# MAGIC Shows which products appear most frequently in recommendations across all customers.
# MAGIC High-scoring products are popular with many customer segments.
# MAGIC
# MAGIC **Chart 2 — Distribution of Recommendation Scores**
# MAGIC Shows the spread of confidence scores. A right-skewed distribution
# MAGIC means most recommendations are low-confidence (as expected for sparse data)
# MAGIC with a few very high-confidence recommendations.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Module 1: Recommendation System — Business Insights',
    fontsize=14, fontweight='bold'
)

# ── Chart 1 — Top 15 Most Recommended Products ────────────
top_products = all_recs_df.groupby('StockCode')['Score'] \
    .sum().sort_values(ascending=False).head(15)
product_desc = retail_clean[['StockCode','Description']] \
    .drop_duplicates('StockCode').set_index('StockCode')['Description']
labels = [
    str(product_desc.get(p, p))[:32] + '...'
    if len(str(product_desc.get(p, p))) > 32
    else str(product_desc.get(p, p))
    for p in top_products.index
]

axes[0].barh(range(len(top_products)), top_products.values,
             color='#2196F3', edgecolor='white', height=0.7)
axes[0].set_yticks(range(len(top_products)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_title('Top 15 Most Recommended Products', fontweight='bold', fontsize=11)
axes[0].set_xlabel('Total Recommendation Score (across all customers)')
axes[0].set_facecolor('#FAFAFA')
axes[0].grid(axis='x', alpha=0.3)

# ── Chart 2 — Score Distribution ──────────────────────────
axes[1].hist(all_recs_df['Score'], bins=50, color='#4CAF50', edgecolor='white')
axes[1].axvline(all_recs_df['Score'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean = {all_recs_df['Score'].mean():.2f}")
axes[1].set_title('Distribution of Recommendation Scores', fontweight='bold', fontsize=11)
axes[1].set_xlabel('Recommendation Score (higher = more confident)')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=10)
axes[1].set_facecolor('#FAFAFA')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/recommendation_charts.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 10 — Save Results to S3
# MAGIC
# MAGIC Saving all recommendations to the S3 `outputs/` folder.
# MAGIC This file is used by Notebook 05 (Large Scale Mining) to build
# MAGIC the master customer dataset, and by the Dashboard for visualisation.

# COMMAND ----------

print("Saving recommendation results to S3...")
print()

save_parquet(all_recs_df, "recommendations")

# Verify
print()
print("Files in S3 outputs/ folder:")
list_files("outputs/")

print()
print("=" * 60)
print("NOTEBOOK 02 — RECOMMENDATION SYSTEM COMPLETE")
print("=" * 60)
print(f"Customers served          : {customer_product.shape[0]:,}")
print(f"Total recommendations     : {len(all_recs_df):,}")
print(f"Catalog coverage          : {coverage:.1f}%")
print(f"Avg similarity score      : {mean_similarity:.4f}")
print(f"MLflow run ID             : {run_id}")
print()
print(f"Top recommended product   : {labels[0]}")
print()
print("Next Step: Run Notebook 03 — Survival Analysis")
