"""
Week 4 Project - System 3: Customer Segmentation
UCT / upskill Campus Internship - Deelipan

Target: 8 behavioral clusters, silhouette score 0.62 (K-means)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv("data/customer_behavior.csv")

behavior_features = [
    "recency_days", "frequency_purchases", "monetary_value",
    "avg_order_value", "tenure_months", "engagement_score"
]
behavior_features = [f for f in behavior_features if f in df.columns]

X = df[behavior_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# 2. OPTIMAL K SELECTION — silhouette score sweep
# ---------------------------------------------------------------------
silhouette_scores = {}
for k in range(2, 12):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = score
    print(f"k={k}: silhouette={score:.3f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest k by silhouette score: {best_k} (score={silhouette_scores[best_k]:.3f})")

plt.figure(figsize=(7, 4))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), "o-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Cluster Count")
plt.tight_layout()
plt.savefig("outputs/figures/silhouette_scores.png")
plt.close()

# ---------------------------------------------------------------------
# 3. FINAL MODEL — 8 clusters (per report)
# ---------------------------------------------------------------------
K_FINAL = 8
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, df["segment"])
print(f"\nFinal model — k={K_FINAL}, silhouette score: {final_silhouette:.3f}")

# ---------------------------------------------------------------------
# 4. SEGMENT PROFILING
# ---------------------------------------------------------------------
segment_profile = df.groupby("segment")[behavior_features].mean()
segment_profile["count"] = df.groupby("segment").size()
print("\nSegment Profiles:\n", segment_profile)

# ---------------------------------------------------------------------
# 5. VISUALIZATION — PCA 2D projection of segments
# ---------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["segment"], cmap="tab10", alpha=0.6)
plt.colorbar(scatter, label="Segment")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"Customer Segments (k={K_FINAL}) — PCA Projection")
plt.tight_layout()
plt.savefig("outputs/figures/customer_segments_pca.png")
plt.close()

# ---------------------------------------------------------------------
# 6. EXPORT SEGMENT ASSIGNMENTS FOR MARKETING USE
# ---------------------------------------------------------------------
df[["segment"] + behavior_features].to_csv("outputs/customer_segments.csv", index=False)

print("\nSegmentation pipeline complete. Results saved to outputs/customer_segments.csv")
