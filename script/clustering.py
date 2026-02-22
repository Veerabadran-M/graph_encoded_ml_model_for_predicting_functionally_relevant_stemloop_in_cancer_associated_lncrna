from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from pathlib import Path


def clustering(embedding_path, n_clusters=2):

    # Load embeddings
    df = pd.read_excel(embedding_path)

    df["embedded_vector"] = df["embedded_vector"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
    )

    X = np.stack(df["embedded_vector"].values)

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # PCA Projection (No Clustering)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c="teal", alpha=0.7)
    plt.title("PCA Projection of Stem-loop Embeddings")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "pca_projection.png", dpi=300)
    plt.close()

    # KMeans on PCA space
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_pca = kmeans_pca.fit_predict(X_pca)

    df["cluster_pcaxkmeans"] = [
        f"Group {c+1}" for c in clusters_pca
    ]

    plt.figure(figsize=(8, 6))
    colors = ["red", "blue"]

    for cluster in range(n_clusters):
        plt.scatter(
            X_pca[clusters_pca == cluster, 0],
            X_pca[clusters_pca == cluster, 1],
            c=colors[cluster],
            label=f"Group {cluster+1}",
            alpha=0.7
        )

    plt.title("KMeans Clustering on PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "kmeans_on_pca.png", dpi=300)
    plt.close()

    # KMeans on Original Embedding
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_original = kmeans_original.fit_predict(X)

    df["cluster_kmeans"] = [
        f"Group {c+1}" for c in clusters_original
    ]

    # t-SNE Projection of Original Embedding
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for cluster in range(n_clusters):
        plt.scatter(
            X_tsne[clusters_original == cluster, 0],
            X_tsne[clusters_original == cluster, 1],
            c=colors[cluster],
            label=f"Group {cluster+1}",
            alpha=0.7
        )

    plt.title("KMeans Clustering on Original Embedding (t-SNE Projection)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "kmeans_on_original_embedding.png", dpi=300)
    plt.close()

    # Export Results
    export_df = df[[
        "symbol",
        "id",
        "cancer_association",
        "cluster_pcaxkmeans",
        "cluster_kmeans"
    ]]

    export_df.to_excel(
        results_dir / "clustering_results.xlsx",
        index=False
    )

    print("Clustering completed.")
    print("Results saved in:", results_dir)