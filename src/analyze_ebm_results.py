#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 explain_seal_gam.py 生成的结果文件，做以下事情：
1) 汇总 27 个网络的 EBM 全局重要性，输出 LaTeX 表格（平均 importance + rank）；
2) 可选：画某个数据集的 shape function 图；
3) 可选：从样本文件中抽取 case study，打印表格内容。

使用前提：
- 每个数据集都已跑过新版本 explain_seal_gam.py，生成了：
  - {dataset}_ebm_estimate_seal_global_{split}.npz
  - {dataset}_ebm_estimate_label_global_{split}.npz
  - {dataset}_ebm_samples_{split}.npz
"""

import os
import glob
import numpy as np
import pandas as pd
import textwrap

import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "n_sub",
    "m_sub",
    "avg_deg",
    "avg_clust",
    "deg_u",
    "deg_v",
    "deg_min",
    "deg_max",
    "deg_sum",
    "deg_prod",
    "CN",
    "AA",
    "RA",
    "Jaccard",
    "sp_len",
    "tri_u",
    "tri_v",
    "tri_uv",
]


def discover_datasets(result_dir: str = ".", split: str = "test"):
    """
    根据 *_ebm_estimate_label_global_{split}.npz 自动发现 dataset_name 列表。
    默认你在当前目录下运行，所有结果文件在同一目录。
    """
    pattern = os.path.join(result_dir, f"*ebm_estimate_label_global_{split}.npz")
    files = glob.glob(pattern)
    datasets = []
    for f in files:
        base = os.path.basename(f)
        # 例：dataset_ebm_estimate_label_global_test.npz
        # 去掉后缀 + 后缀名
        prefix = f"_ebm_estimate_label_global_{split}.npz"
        if base.endswith(prefix):
            ds = base[: -len(prefix)]
            datasets.append(ds)
    datasets = sorted(list(set(datasets)))
    return datasets


def load_global_importance_for_dataset(dataset: str,
                                       split: str = "test",
                                       result_dir: str = "."):
    """
    加载一个数据集的 SEAL-EBM 和 Label-EBM 的 global importance.
    返回两个 dict[feature] = importance_value
    """
    seal_path = os.path.join(result_dir,
                             f"{dataset}_ebm_estimate_seal_global_{split}.npz")
    label_path = os.path.join(result_dir,
                              f"{dataset}_ebm_estimate_label_global_{split}.npz")

    if not os.path.exists(seal_path):
        raise FileNotFoundError(seal_path)
    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    d_seal = np.load(seal_path)
    d_label = np.load(label_path)

    seal_imp = {}
    label_imp = {}
    for f in FEATURE_NAMES:
        k_s = f"importance_{f}"
        if k_s in d_seal:
            seal_imp[f] = float(d_seal[k_s][0])
        else:
            seal_imp[f] = 0.0  # 如果缺失，设为 0
        k_l = f"importance_{f}"
        if k_l in d_label:
            label_imp[f] = float(d_label[k_l][0])
        else:
            label_imp[f] = 0.0

    return seal_imp, label_imp


def aggregate_importance(result_dir: str = ".", split: str = "test"):
    """
    跨所有数据集汇总 importance:
    返回一个 DataFrame，包含每个特征的 mean/std (label/seal) 和 mean rank。
    """
    datasets = discover_datasets(result_dir=result_dir, split=split)
    if len(datasets) == 0:
        raise RuntimeError(
            f"在 {result_dir} 下未找到 '*_ebm_estimate_label_global_{split}.npz' 文件")

    print(f"发现 {len(datasets)} 个数据集: {datasets}")

    # 收集 importance 矩阵
    label_mat = []  # shape: [n_dataset, n_feature]
    seal_mat = []
    for ds in datasets:
        seal_imp, label_imp = load_global_importance_for_dataset(
            ds, split=split, result_dir=result_dir
        )
        label_mat.append([label_imp[f] for f in FEATURE_NAMES])
        seal_mat.append([seal_imp[f] for f in FEATURE_NAMES])

    label_mat = np.array(label_mat)  # [D, F]
    seal_mat = np.array(seal_mat)

    # mean / std
    label_mean = label_mat.mean(axis=0)
    label_std = label_mat.std(axis=0)
    seal_mean = seal_mat.mean(axis=0)
    seal_std = seal_mat.std(axis=0)

    # rank：对每个数据集分别排序，然后对 rank 取平均
    # 这里 rank=1 表示最重要（值最大）
    n_ds, n_feat = label_mat.shape
    label_ranks = np.zeros_like(label_mat, dtype=float)
    seal_ranks = np.zeros_like(seal_mat, dtype=float)

    for i in range(n_ds):
        # argsort -> 0 为最小; 我们要 1 为最大
        order_l = np.argsort(label_mat[i])          # 0..F-1 (ascending)
        inv_order_l = np.empty_like(order_l)
        inv_order_l[order_l] = np.arange(n_feat)    # 0..F-1 (rank index)
        ranks_l = n_feat - inv_order_l              # F..1
        label_ranks[i] = ranks_l

        order_s = np.argsort(seal_mat[i])
        inv_order_s = np.empty_like(order_s)
        inv_order_s[order_s] = np.arange(n_feat)
        ranks_s = n_feat - inv_order_s
        seal_ranks[i] = ranks_s

    label_rank_mean = label_ranks.mean(axis=0)
    seal_rank_mean = seal_ranks.mean(axis=0)

    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "label_mean": label_mean,
        "label_std": label_std,
        "seal_mean": seal_mean,
        "seal_std": seal_std,
        "label_rank_mean": label_rank_mean,
        "seal_rank_mean": seal_rank_mean,
    })

    return df, datasets


def df_to_latex_table(df: pd.DataFrame,
                      float_fmt: str = "{:.3f}",
                      caption: str | None = None,
                      label: str | None = None) -> str:
    """
    把 aggregate_importance 得到的 df 转换为 LaTeX 表格字符串。
    """
    if caption is None:
        caption = ("Global importance statistics of structural features across all datasets. "
                   "For each feature, we show mean and standard deviation of the global "
                   "importance assigned by Label--EBM and SEAL--EBM, as well as average ranks.")

    if label is None:
        label = "tab:ebm_global_importance_stats"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append("  \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append("    \\multirow{2}{*}{Feature} &")
    lines.append("    \\multicolumn{2}{c}{Label--EBM importance} &")
    lines.append("    \\multicolumn{2}{c}{SEAL--EBM importance} &")
    lines.append("    \\multicolumn{2}{c}{Avg. rank} \\\\")
    lines.append("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    lines.append("     & mean & std & mean & std & Label & SEAL \\\\")
    lines.append("    \\midrule")

    for _, row in df.iterrows():
        feat = row["feature"]
        # 对 feature 名做一点 LaTeX 友好处理
        feat_tex = feat.replace("_", "\\_")
        l_m = float_fmt.format(row["label_mean"])
        l_s = float_fmt.format(row["label_std"])
        s_m = float_fmt.format(row["seal_mean"])
        s_s = float_fmt.format(row["seal_std"])
        r_l = float_fmt.format(row["label_rank_mean"])
        r_s = float_fmt.format(row["seal_rank_mean"])
        line = f"    {feat_tex:10s} & {l_m} & {l_s} & {s_m} & {s_s} & {r_l} & {r_s} \\\\"
        lines.append(line)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


########################
# shape function 绘图
########################

def plot_shape_functions(dataset: str,
                         result_dir: str = ".",
                         split: str = "test",
                         features_to_plot=None,
                         out_prefix: str | None = None):
    """
    为指定数据集画若干特征的 shape function（Label-EBM 与 SEAL-EBM 对比）。
    依赖 explain_seal_gam.py 中 export_ebm_global 保存的 grid_*, shape_*。
    """
    if features_to_plot is None:
        # 默认画四个最关键特征
        features_to_plot = ["sp_len", "Jaccard", "tri_uv", "deg_u"]

    seal_path = os.path.join(result_dir,
                             f"{dataset}_ebm_estimate_seal_global_{split}.npz")
    label_path = os.path.join(result_dir,
                              f"{dataset}_ebm_estimate_label_global_{split}.npz")
    if not (os.path.exists(seal_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f"缺少 global npz: {seal_path} 或 {label_path}")

    d_seal = np.load(seal_path)
    d_label = np.load(label_path)

    n_feat = len(features_to_plot)
    plt.figure(figsize=(4 * n_feat, 6))

    for i, feat in enumerate(features_to_plot):
        # 读取 grid 和 shape
        g_key = f"grid_{feat}"
        s_key = f"shape_{feat}"

        # 对 Label-EBM
        if g_key in d_label and s_key in d_label:
            x_l = d_label[g_key]
            y_l = d_label[s_key]
        else:
            x_l, y_l = None, None

        # 对 SEAL-EBM
        if g_key in d_seal and s_key in d_seal:
            x_s = d_seal[g_key]
            y_s = d_seal[s_key]
        else:
            x_s, y_s = None, None

        plt.subplot(2, n_feat, i + 1)
        if x_l is not None and y_l is not None:
            plt.plot(x_l, y_l, marker="o")
        plt.title(f"Label-EBM: {feat}")
        plt.xlabel(feat)
        plt.ylabel("contribution (log-odds or score)")

        plt.subplot(2, n_feat, n_feat + i + 1)
        if x_s is not None and y_s is not None:
            plt.plot(x_s, y_s, marker="o", color="C1")
        plt.title(f"SEAL-EBM: {feat}")
        plt.xlabel(feat)
        plt.ylabel("contribution (log-odds or score)")

    plt.tight_layout()

    if out_prefix is None:
        out_prefix = f"{dataset}_shapes_{split}"

    out_pdf = f"{out_prefix}.pdf"
    out_png = f"{out_prefix}.png"
    plt.savefig(out_pdf, bbox_inches="tight",dpi=300)
    plt.savefig(out_png, bbox_inches="tight",dpi=300)
    plt.close()
    print(f"shape function 图已保存到: {out_pdf}, {out_png}")


########################
# case study
########################

def load_samples_for_dataset(dataset: str,
                             result_dir: str = ".",
                             split: str = "test"):
    """
    加载 explain_seal_gam.py 保存的 *_ebm_samples_{split}.npz：
      - X
      - y_true
      - y_seal
      - y_seal_pred
      - y_ebm_label_proba
      - reg_metrics
      - clf_metrics
    """
    path = os.path.join(result_dir, f"{dataset}_ebm_samples_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y_true = data["y_true"]
    y_seal = data["y_seal"]
    y_seal_pred = data["y_seal_pred"]
    y_ebm_label_proba = data["y_ebm_label_proba"]
    reg_metrics = data["reg_metrics"].item()
    clf_metrics = data["clf_metrics"].item()
    return X, y_true, y_seal, y_seal_pred, y_ebm_label_proba, reg_metrics, clf_metrics


def select_case_studies(dataset: str,
                        result_dir: str = ".",
                        split: str = "test",
                        num_per_type: int = 2,
                        random_state: int = 0):
    """
    简单策略：
      - 在该数据集下，从样本中选取若干:
        - TP: y_true=1 且 y_ebm_label_proba 高
        - FP: y_true=0 且 y_ebm_label_proba 高
        - FN: y_true=1 且 y_ebm_label_proba 低
    输出 info，并打印成表格形式（不含 local decomposition）。
    如需 local decomposition，可在 notebook 中加载 local npz 再 join。
    """
    rng = np.random.RandomState(random_state)
    X, y_true, y_seal, y_seal_pred, y_ebm_label_proba, reg_metrics, clf_metrics = \
        load_samples_for_dataset(dataset, result_dir=result_dir, split=split)

    # 转为 DataFrame 方便排序
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["y_true"] = y_true
    df["y_seal"] = y_seal
    df["y_seal_pred"] = y_seal_pred
    df["y_ebm_label_proba"] = y_ebm_label_proba

    # True positives: label=1, proba 高
    tps = df[df["y_true"] == 1].sort_values("y_ebm_label_proba", ascending=False)
    fps = df[df["y_true"] == 0].sort_values("y_ebm_label_proba", ascending=False)
    fns = df[df["y_true"] == 1].sort_values("y_ebm_label_proba", ascending=True)

    # 随机挑 num_per_type 个（如果少于 num_per_type 就全取）
    def sample_rows(dsub):
        if len(dsub) <= num_per_type:
            return dsub
        idx = rng.choice(len(dsub), size=num_per_type, replace=False)
        return dsub.iloc[idx]

    tps_s = sample_rows(tps)
    fps_s = sample_rows(fps)
    fns_s = sample_rows(fns)

    tps_s["case_type"] = "TP"
    fps_s["case_type"] = "FP"
    fns_s["case_type"] = "FN"

    cases = pd.concat([tps_s, fps_s, fns_s], axis=0)
    cases = cases.reset_index(drop=True)

    # 打印为 markdown 表（你可以直接 copy 到 LaTeX 或再转 LaTeX）
    print("\nCase study (without local decomposition):")
    print(cases[["case_type", "y_true", "y_ebm_label_proba", "y_seal",
                 "n_sub", "m_sub", "avg_deg", "avg_clust",
                 "deg_u", "deg_v", "CN", "AA", "RA", "Jaccard", "sp_len", "tri_uv"]].to_markdown(index=False))

    return cases


########################
# main 入口
########################

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default=".",
                        help="存放 *_ebm_estimate_*_global_*.npz 和 *_ebm_samples_*.npz 的目录")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])

    parser.add_argument("--do_table", action="store_true",
                        help="汇总所有数据集的全局重要性并输出 LaTeX 表")
    parser.add_argument("--do_shapes", action="store_true",
                        help="为指定数据集画 shape function 图")
    parser.add_argument("--do_cases", action="store_true",
                        help="为指定数据集选取 case study 并打印表格")

    parser.add_argument("--dataset_for_shapes", type=str, default=None,
                        help="画 shape 的数据集名（不含后缀），例如 'FBK'")
    parser.add_argument("--dataset_for_cases", type=str, default=None,
                        help="做 case study 的数据集名，例如 'FBK'")
    parser.add_argument("--features_for_shapes", type=str, nargs="*",
                        default=None,
                        help="指定要画 shape 的特征名列表，如 --features_for_shapes sp_len Jaccard tri_uv deg_u")

    args = parser.parse_args()

    if args.do_table:
        df, datasets = aggregate_importance(result_dir=args.result_dir, split=args.split)
        # 排序：按 label_mean 从大到小
        df_sorted = df.sort_values("label_mean", ascending=False).reset_index(drop=True)
        latex_str = df_to_latex_table(df_sorted)
        print("\n================ LaTeX 表格 =================\n")
        print(latex_str)
        print("\n=============================================\n")

        # 你也可以保存到文件
        out_path = os.path.join(args.result_dir, f"ebm_global_importance_stats_{args.split}.tex")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"LaTeX 表格已输出到: {out_path}")

    if args.do_shapes:
        if args.dataset_for_shapes is None:
            raise ValueError("--do_shapes 需要指定 --dataset_for_shapes")
        plot_shape_functions(
            args.dataset_for_shapes,
            result_dir=args.result_dir,
            split=args.split,
            features_to_plot=args.features_for_shapes,
        )

    if args.do_cases:
        if args.dataset_for_cases is None:
            raise ValueError("--do_cases 需要指定 --dataset_for_cases")
        _ = select_case_studies(
            args.dataset_for_cases,
            result_dir=args.result_dir,
            split=args.split,
        )


if __name__ == "__main__":
    main()
