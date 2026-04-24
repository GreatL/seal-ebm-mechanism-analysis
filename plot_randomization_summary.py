#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 与 run_randomization_batch.py 中保持一致
FEATURE_LIST = [
    "deg_u","deg_v","deg_min","deg_max","deg_sum","deg_prod",
    "CN","AA","RA","Jaccard","sp_len","tri_u","tri_v","tri_uv",
]

KEY_FEATS = ["tri_uv", "Jaccard", "RA", "CN", "sp_len", "deg_u", "deg_prod"]

def plot_bar_per_dataset(df_all, out_dir="fig_randomization_bar"):
    os.makedirs(out_dir, exist_ok=True)

    datasets = sorted(df_all["Dataset"].unique())
    for ds in datasets:
        df_ds = df_all[df_all["Dataset"] == ds].copy()
        # 按 FEATURE_LIST 的顺序排序
        df_ds["feat_order"] = df_ds["Feature"].apply(lambda f: FEATURE_LIST.index(f))
        df_ds = df_ds.sort_values("feat_order")

        feats = df_ds["Feature"].tolist()
        imp_orig = df_ds["Importance_orig"].values
        imp_rand = df_ds["Importance_rand"].values

        x = np.arange(len(feats))
        width = 0.35

        plt.figure(figsize=(max(6, len(feats)*0.45), 4))
        plt.bar(x - width/2, imp_orig, width, label="Original")
        plt.bar(x + width/2, imp_rand, width, label="Randomized")

        plt.xticks(x, feats, rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(f"Feature importance: original vs randomized ({ds})")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"rand_bar_{ds}.png")
        out_pdf = os.path.join(out_dir, f"rand_bar_{ds}.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf, dpi=300)
        plt.close()
        print(f"[SAVE] {out_png}")

def plot_lines_key_feats(df_all, out_dir="fig_randomization_lines"):
    os.makedirs(out_dir, exist_ok=True)

    datasets = sorted(df_all["Dataset"].unique())
    # 只画 KEY_FEATS 的折线图，便于突出 triangle / similarity / degree
    feats = [f for f in KEY_FEATS if f in df_all["Feature"].unique()]
    if not feats:
        print("[WARN] none of KEY_FEATS present in data, skip line plots.")
        return

    x = np.arange(len(feats))

    for ds in datasets:
        df_ds = df_all[(df_all["Dataset"] == ds) &
                       (df_all["Feature"].isin(feats))].copy()
        df_ds["feat_order"] = df_ds["Feature"].apply(lambda f: feats.index(f))
        df_ds = df_ds.sort_values("feat_order")

        imp_orig = df_ds["Importance_orig"].values
        imp_rand = df_ds["Importance_rand"].values

        plt.figure(figsize=(max(6, len(feats)*0.6), 4))
        plt.plot(x, imp_orig, marker="o", label="Original")
        plt.plot(x, imp_rand, marker="s", label="Randomized")

        plt.xticks(x, feats, rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(f"Key feature importance: original vs randomized ({ds})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"rand_lines_{ds}.png")
        out_pdf = os.path.join(out_dir, f"rand_lines_{ds}.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf, dpi=300)
        plt.close()
        print(f"[SAVE] {out_png}")

def main():
    df_all = pd.read_csv("randomization_importances.csv")
    print("[INFO] loaded randomization_importances.csv")
    print(df_all.head())

    # 画每个数据集的全特征柱状图
    plot_bar_per_dataset(df_all)

    # 画每个数据集的 key features 折线图
    plot_lines_key_feats(df_all)

    print("[DONE] all figures generated.")

if __name__ == "__main__":
    main()
