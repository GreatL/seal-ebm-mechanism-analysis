#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

FEATURE_ORDER = [
    "n_sub","m_sub","avg_deg","avg_clust",
    "deg_u","deg_v","deg_min","deg_max","deg_sum","deg_prod",
    "CN","AA","RA","Jaccard","sp_len","tri_u","tri_v","tri_uv",
]

NETWORK_TYPES = {
    "ADV": "Social-like",
    "BUP": "Social-like",
    "CDM": "Collaboration",
    "Celegans": "Biological",
    "CGS": "Collaboration",
    "ecoli": "Biological",
    "EML": "Social-like",
    "ERD": "Collaboration",
    "FBK": "Social-like",
    "GRQ": "Collaboration",
    "HMT": "Social-like",
    "HPD": "Biological",
    "HTC": "Collaboration",
    "INF": "Social-like",
    "KHN": "Collaboration",
    "LDG": "Collaboration",
    "NS": "Collaboration",
    "NSC": "Collaboration",
    "PB": "Social-like",
    "PGP": "Social-like",
    "Power": "Technological",
    "Router": "Technological",
    "SMG": "Collaboration",
    "USAir": "Technological",
    "Yeast": "Biological",
    "YST": "Biological",
    "ZWL": "Collaboration",
}

def main():
    df_long = pd.read_csv("ebm_feature_importance_all_wide.csv")

    # 透视：每个 Dataset 一行，每个 Feature 一列（label_test）
    df_label = df_long.pivot_table(
        index="Dataset",
        columns="Feature",
        values="label_test" #label
        #values="seal_test" #seal
    )

    # 保证列按 FEATURE_ORDER 顺序，且只保留 CSV 中确实存在的特征
    available_feats = [f for f in FEATURE_ORDER if f in df_label.columns]
    df_label = df_label[available_feats].copy()

    print("[INFO] features used for PCA/t-SNE:", available_feats)

    X = df_label.values
    datasets = df_label.index.to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=10, learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled)

    type_colors = {
        "Social-like": "tab:blue",
        "Collaboration": "tab:green",
        "Biological": "tab:red",
        "Technological": "tab:orange",
        "Unknown": "k",
    }

    types = [NETWORK_TYPES.get(ds, "Unknown") for ds in datasets]

    # PCA 图
    plt.figure(figsize=(6,5))
    for t in sorted(set(types)):
        mask = [tt == t for tt in types]
        X_sub = X_pca[mask]
        plt.scatter(X_sub[:,0], X_sub[:,1],
                    c=type_colors.get(t,"k"), label=t, s=40, alpha=0.8)
    for (x, y, ds) in zip(X_pca[:,0], X_pca[:,1], datasets):
        plt.text(x, y, ds, fontsize=7, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Label-EBM importance (ensemble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ensemble_pca_label_long.pdf", dpi=300)
    plt.savefig("ensemble_pca_label_long.png", dpi=300)

    # t-SNE 图
    plt.figure(figsize=(6,5))
    for t in sorted(set(types)):
        mask = [tt == t for tt in types]
        X_sub = X_tsne[mask]
        plt.scatter(X_sub[:,0], X_sub[:,1],
                    c=type_colors.get(t,"k"), label=t, s=40, alpha=0.8)
    for (x, y, ds) in zip(X_tsne[:,0], X_tsne[:,1], datasets):
        plt.text(x, y, ds, fontsize=7, alpha=0.7)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Label-EBM importance (ensemble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ensemble_tsne_label_long.pdf", dpi=300)
    plt.savefig("ensemble_tsne_label_long.png", dpi=300)

    print("[DONE] ensemble PCA/t-SNE saved.")

if __name__ == "__main__":
    main()
