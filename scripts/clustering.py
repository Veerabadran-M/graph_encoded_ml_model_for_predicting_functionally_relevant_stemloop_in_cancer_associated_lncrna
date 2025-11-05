from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

path = "embedded vector excel path"

df = pd.read_excel(path)

df['embedded_vector'] = df['embedded_vector'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

X = np.stack(df['embedded_vector'].values)
ids = df['id'].tolist()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='teal', alpha=0.7)
plt.title("PCA Projection of Stem-loop Embeddings")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

df['cluster'] = clusters

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
colors = ['red', 'blue']
for cluster in [0,1]:
    plt.scatter(X_tsne[clusters==cluster,0],
                X_tsne[clusters==cluster,1],
                c=colors[cluster],
                label=f'Cluster {cluster}', alpha=0.6)

plt.title("Stem-loop Clusters (t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()

cancer_path = 'cancer stemloop data excel path'
non_cancer_path = 'non ancer stemloop data excel path'

cdf = pd.read_excel(cancer_path)
ncdf = pd.read_excel(non_cancer_path)

output_df = pd.DataFrame({
    'Stemloop_id': df['id'],
    'cluster': clusters
})

df_merged = output_df.merge(cdf, on='Stemloop_id', how='left')

df_merged = df_merged.merge(ncdf, on='Stemloop_id', how='left')

df_merged.to_excel("desired output path", index=False)