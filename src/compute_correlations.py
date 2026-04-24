#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import spearmanr, pearsonr

FEATURE_TRI = "tri_uv"
FEATURE_DEGU = "deg_u"

def main():
    # 读入长表 importance 和网络全局统计
    df_imp = pd.read_csv("ebm_feature_importance_all_wide.csv")
    df_stats = pd.read_csv("network_stats.csv")

    # 先透视成每个 Dataset 一行的宽表（只用 label_test）
    df_pivot = df_imp.pivot_table(
        index="Dataset",
        columns="Feature",
        values="label_test"
    )

    # 合并
    df = df_pivot.reset_index()
    df = pd.merge(df, df_stats, on="Dataset", how="inner")

    print("[INFO] merged dataframe columns:", df.columns.tolist())

    # -------- tri_uv importance vs global_clustering --------
    if FEATURE_TRI in df.columns:
        x = df["global_clustering"].values
        y = df[FEATURE_TRI].values
        valid = ~pd.isna(x) & ~pd.isna(y)
        xv, yv = x[valid], y[valid]

        print("\n[tri_uv vs global_clustering]")
        print("Datasets used:", df["Dataset"][valid].tolist())

        rho_s, p_s = spearmanr(xv, yv)
        rho_p, p_p = pearsonr(xv, yv)
        print(f"Spearman  ρ = {rho_s:.3f}, p = {p_s:.3e}")
        print(f"Pearson   r = {rho_p:.3f}, p = {p_p:.3e}")
    else:
        print(f"[WARN] {FEATURE_TRI} not found in importance table; skip tri_uv correlation.")

    # -------- deg_u importance vs degree_std --------
    if FEATURE_DEGU in df.columns:
        x = df["degree_std"].values
        y = df[FEATURE_DEGU].values
        valid = ~pd.isna(x) & ~pd.isna(y)
        xv, yv = x[valid], y[valid]

        print("\n[deg_u vs degree_std]")
        print("Datasets used:", df["Dataset"][valid].tolist())

        rho_s, p_s = spearmanr(xv, yv)
        rho_p, p_p = pearsonr(xv, yv)
        print(f"Spearman  ρ = {rho_s:.3f}, p = {p_s:.3e}")
        print(f"Pearson   r = {rho_p:.3f}, p = {p_p:.3e}")
    else:
        print(f"[WARN] {FEATURE_DEGU} not found in importance table; skip deg_u correlation.")

if __name__ == "__main__":
    main()
