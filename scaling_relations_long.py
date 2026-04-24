#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df_imp_long = pd.read_csv("ebm_feature_importance_all_wide.csv")
    df_stats = pd.read_csv("network_stats.csv")

    # 先把长表透视成：每个 Dataset 一行，列为 tri_uv 和 deg_u 的 label_test
    df_pivot = df_imp_long.pivot_table(
        index="Dataset",
        columns="Feature",
        values="label_test"
    )

    # 检查一下有哪些特征列
    print("[INFO] available features:", df_pivot.columns.tolist())

    needed = []
    for f in ["tri_uv", "deg_u"]:
        if f not in df_pivot.columns:
            print(f"[WARN] feature '{f}' not found in importance table.")
        else:
            needed.append(f)
    if len(needed) == 0:
        print("[ERROR] neither tri_uv nor deg_u found, abort.")
        return

    df_imp = df_pivot.reset_index()  # Dataset 变回普通列
    df = pd.merge(df_imp, df_stats, on="Dataset", how="inner")

    # tri_uv importance vs global clustering
    if "tri_uv" in df.columns:
        tri_imp = df["tri_uv"].values
        glob_clust = df["global_clustering"].values
        datasets = df["Dataset"].values

        plt.figure(figsize=(5,4))
        plt.scatter(glob_clust, tri_imp, c="tab:blue")
        for x, y, ds in zip(glob_clust, tri_imp, datasets):
            plt.text(x, y, ds, fontsize=7, alpha=0.7)
        plt.xlabel("Global clustering coefficient")
        plt.ylabel("Label-EBM importance of tri_uv")
        plt.title("tri_uv importance vs global clustering")
        plt.tight_layout()
        plt.savefig("scaling_triuv_vs_clustering.pdf", dpi=300)
        plt.savefig("scaling_triuv_vs_clustering.png", dpi=300)
        print("[DONE] tri_uv scaling plot saved.")
    else:
        print("[WARN] tri_uv not present, skip tri_uv plot.")

    # deg_u importance vs degree heterogeneity (std of degree)
    if "deg_u" in df.columns:
        deg_imp = df["deg_u"].values
        deg_std = df["degree_std"].values
        datasets = df["Dataset"].values

        plt.figure(figsize=(5,4))
        plt.scatter(deg_std, deg_imp, c="tab:red")
        for x, y, ds in zip(deg_std, deg_imp, datasets):
            plt.text(x, y, ds, fontsize=7, alpha=0.7)
        plt.xlabel("Degree standard deviation")
        plt.ylabel("Label-EBM importance of deg_u")
        plt.title("deg_u importance vs degree heterogeneity")
        plt.tight_layout()
        plt.savefig("scaling_degu_vs_degstd.pdf", dpi=300)
        plt.savefig("scaling_degu_vs_degstd.png", dpi=300)
        print("[DONE] deg_u scaling plot saved.")
    else:
        print("[WARN] deg_u not present, skip deg_u plot.")

if __name__ == "__main__":
    main()
