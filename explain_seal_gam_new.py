#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import scipy.sparse as ssp
import networkx as nx

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)

from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)

from seal_from_edgelist import (
    DGCNN,
    get_cache_path,
    get_model_path,
    get_dataset_name,
)

########################
# 结构特征
########################

def compute_struct_features(sub_adj: ssp.csr_matrix,
                            src: int = 0,
                            dst: int = 1) -> np.ndarray:
    """对一个子图邻接矩阵 sub_adj 计算一组简单的结构特征."""
    G = nx.from_scipy_sparse_array(sub_adj)

    n_sub = G.number_of_nodes()
    m_sub = G.number_of_edges()
    avg_deg = 2 * m_sub / n_sub if n_sub > 0 else 0.0

    try:
        avg_clust = nx.average_clustering(G)
    except Exception:
        avg_clust = 0.0

    deg_u = G.degree[src] if src in G else 0
    deg_v = G.degree[dst] if dst in G else 0

    deg_min = min(deg_u, deg_v)
    deg_max = max(deg_u, deg_v)
    deg_sum = deg_u + deg_v
    deg_prod = deg_u * deg_v

    if src in G and dst in G:
        cn_nodes = list(nx.common_neighbors(G, src, dst))
    else:
        cn_nodes = []

    cn = len(cn_nodes)
    aa = 0.0
    ra = 0.0
    for w in cn_nodes:
        dw = G.degree[w]
        if dw > 1:
            aa += 1.0 / np.log(dw + 1e-8)
            ra += 1.0 / dw

    try:
        neigh_u = set(G.neighbors(src)) if src in G else set()
        neigh_v = set(G.neighbors(dst)) if dst in G else set()
        inter = len(neigh_u & neigh_v)
        union = len(neigh_u | neigh_v)
        jaccard = inter / union if union > 0 else 0.0
    except Exception:
        jaccard = 0.0

    try:
        sp_len = nx.shortest_path_length(G, source=src, target=dst)
    except nx.NetworkXNoPath:
        sp_len = 0

    try:
        triangles = nx.triangles(G)
        tri_u = triangles.get(src, 0)
        tri_v = triangles.get(dst, 0)
        tri_uv = 0
        for w in cn_nodes:
            if G.has_edge(src, w) and G.has_edge(dst, w):
                tri_uv += 1
    except Exception:
        tri_u = tri_v = tri_uv = 0

    feats = np.array([
        n_sub,       # 0
        m_sub,       # 1
        avg_deg,     # 2
        avg_clust,   # 3
        deg_u,       # 4
        deg_v,       # 5
        deg_min,     # 6
        deg_max,     # 7
        deg_sum,     # 8
        deg_prod,    # 9
        cn,          # 10
        aa,          # 11
        ra,          # 12
        jaccard,     # 13
        sp_len,      # 14
        tri_u,       # 15
        tri_v,       # 16
        tri_uv,      # 17
    ], dtype=np.float32)

    return feats


def collect_features_and_seal_outputs(
        cache_path: str,
        model_path: str,
        device: torch.device,
        use_split: str = "test",
        batch_size: int = 64,
        max_samples: int | None = None):
    """
    从 SEAL 缓存 + 模型中提取:
      - X: 子图结构特征
      - y_seal: SEAL 输出概率
      - y_true: 真实标签
    """
    cache = torch.load(cache_path, map_location="cpu")
    train_data = cache["train_data"]
    val_data = cache["val_data"]
    test_data = cache["test_data"]

    if use_split == "train":
        data_list = train_data
    elif use_split == "valid":
        data_list = val_data
    else:
        data_list = test_data

    ckpt = torch.load(model_path, map_location=device)
    model_args = ckpt.get("args", None)

    hidden = model_args["hidden"] if model_args is not None and "hidden" in model_args else 32
    layers = model_args["layers"] if model_args is not None and "layers" in model_args else 3
    max_z = 1000
    k = 30

    model = DGCNN(hidden, layers, max_z, k).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_feats = []
    all_y_seal = []
    all_y_true = []

    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            logit = model(d.z, d.edge_index, d.batch)
            prob = torch.sigmoid(logit.view(-1)).cpu().numpy()
            labels = d.y.view(-1).cpu().numpy()

            edge_index = d.edge_index.cpu().numpy()
            batch_vec = d.batch.cpu().numpy()
            num_graphs = d.num_graphs

            for g_idx in range(num_graphs):
                mask = (batch_vec == g_idx)
                nodes_in_g = np.where(mask)[0]
                local_id = {nid: j for j, nid in enumerate(nodes_in_g)}
                e_mask = np.isin(edge_index[0], nodes_in_g)
                e_idx = edge_index[:, e_mask]

                if e_idx[0].size == 0:
                    u = np.array([], dtype=int)
                    v = np.array([], dtype=int)
                else:
                    u = np.array([local_id[x] for x in e_idx[0]], dtype=int)
                    v = np.array([local_id[x] for x in e_idx[1]], dtype=int)

                n_sub = len(nodes_in_g)
                sub_adj = ssp.csr_matrix(
                    (np.ones(len(u), dtype=np.float32), (u, v)),
                    shape=(n_sub, n_sub)
                )
                sub_adj = sub_adj + sub_adj.T
                sub_adj.data[:] = 1.0
                sub_adj.eliminate_zeros()

                feats = compute_struct_features(sub_adj, src=0, dst=1)
                all_feats.append(feats)

            all_y_seal.extend(prob.tolist())
            all_y_true.extend(labels.tolist())

            if max_samples is not None and len(all_feats) >= max_samples:
                break

    X = np.vstack(all_feats)
    y_seal = np.array(all_y_seal[:len(all_feats)], dtype=np.float32)
    y_true = np.array(all_y_true[:len(all_feats)], dtype=np.float32)

    return X, y_seal, y_true


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

########################
# EBM 训练
########################

def train_ebm_estimate_seal(
        X: np.ndarray,
        y_seal: np.ndarray,
        interactions: int = 0,
        random_state: int = 0):
    model = ExplainableBoostingRegressor(
        interactions=interactions,
        max_bins=256,
        learning_rate=0.01,
        random_state=random_state,
    )
    model.fit(X, y_seal)

    y_pred = model.predict(X)
    pearson = np.corrcoef(y_pred, y_seal)[0, 1]
    mse = mean_squared_error(y_seal, y_pred)

    metrics = {
        "pearson": float(pearson),
        "mse": float(mse),
    }
    return model, metrics


def train_ebm_estimate_label(
        X: np.ndarray,
        y_true: np.ndarray,
        interactions: int = 0,
        random_state: int = 0):
    model = ExplainableBoostingClassifier(
        interactions=interactions,
        max_bins=256,
        learning_rate=0.01,
        random_state=random_state,
    )
    model.fit(X, y_true)

    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)

    metrics = {
        "auc": float(auc),
        "ap": float(ap),
    }
    return model, metrics


########################
# 导出特征重要性 + shape
########################

def export_ebm_global(global_exp, feature_names, out_path: str):
    """
    同时导出:
      - importance_{feat}: 全局一元重要性 (scalar)
      - grid_{feat}: 1D shape 的 x 轴 (feature values / bin centers)
      - shape_{feat}: 1D shape 的 y 轴 (log-odds 或 regression contribution)
    """
    data = global_exp.data()
    names = data["names"]           # ['feature_0000', ..., 'feature_0000 & feature_0001', ...]
    scores = data["scores"]         # importance (scalar per feature)
    # 对于 glassbox EBM，display_data 是每个特征的 shape 信息
    display_data = data.get("display_data", None)

    # 只保留一元特征（没有 & 的）
    name2score = {
        n: float(s) for n, s in zip(names, scores)
        if "&" not in n
    }

    out_dict = {}
    for j, fname in enumerate(feature_names):
        internal_name = f"feature_{j:04d}"
        if internal_name in name2score:
            out_dict[f"importance_{fname}"] = np.array([name2score[internal_name]], dtype=float)
        else:
            print(f"[警告] 在 EBM explanation.names 里找不到 {internal_name}，跳过 importance {fname}")

    # 处理 shape function
    if display_data is not None:
        # display_data 是一个 list，每个元素对应一个 feature 的 dict
        for j, fname in enumerate(feature_names):
            internal_name = f"feature_{j:04d}"
            try:
                feat_disp = display_data[j]  # 通常 index 对应 feature_j
            except Exception:
                print(f"[警告] display_data 索引 {j} 不可用，跳过 shape {fname}")
                continue

            # feat_disp 通常包含 'scores' (y) 与 'values' 或 'bins' (x)
            # interpret 版本之间字段名有差异，这里做兼容
            x_vals = None
            y_vals = None

            if isinstance(feat_disp, dict):
                if "scores" in feat_disp:
                    y_vals = np.array(feat_disp["scores"], dtype=float)
                if "values" in feat_disp:
                    x_vals = np.array(feat_disp["values"], dtype=float)
                elif "bins" in feat_disp:
                    x_vals = np.array(feat_disp["bins"], dtype=float)
            if x_vals is None or y_vals is None:
                # 无法解析就跳过
                print(f"[警告] 未能解析 {fname} 的 shape function，跳过 grid/shape 保存")
                continue

            out_dict[f"grid_{fname}"] = x_vals
            out_dict[f"shape_{fname}"] = y_vals

    np.savez(out_path, **out_dict)
    print(f"EBM 全局解释已保存到: {out_path}")


########################
# 主函数
########################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--edge_path", required=True)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--model_name", default=None)

    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--interactions", type=int, default=0)

    # 可选：是否保存全部 local explanation（体积很大，默认 False）
    parser.add_argument("--save_local", action="store_true",
                        help="如果打开，将对所有样本调用 explain_local 并保存局部贡献（注意：文件很大）。")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DummyArgs:
        def __init__(self, edge_path, num_hops, val_ratio, test_ratio, seed,
                     model_dir, model_name):
            self.edge_path = edge_path
            self.num_hops = num_hops
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.seed = seed
            self.model_dir = model_dir
            self.model_name = model_name

    dargs = DummyArgs(
        edge_path=args.edge_path,
        num_hops=args.num_hops,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        model_dir=args.model_dir,
        model_name=args.model_name,
    )

    cache_path = get_cache_path(dargs)
    model_path = get_model_path(dargs)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"缓存文件不存在: {cache_path}\n"
            f"请先运行 seal_from_edgelist.py 生成缓存。"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请先训练并保存模型到 {model_path}。"
        )

    print(f"使用缓存: {cache_path}")
    print(f"使用模型: {model_path}")
    print(f"收集 {args.split} 子图的结构特征、SEAL 输出以及真实标签...")

    X, y_seal, y_true = collect_features_and_seal_outputs(
        cache_path,
        model_path,
        device,
        use_split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
    if X.shape[1] != len(FEATURE_NAMES):
        print("警告：特征维度与 FEATURE_NAMES 长度不一致，请检查。")

    dataset_name = get_dataset_name(args.edge_path)

    print("\n=== 训练 EBM_estimate_SEAL（回归：拟合 SEAL 输出） ===")
    ebm_reg, reg_metrics = train_ebm_estimate_seal(
        X, y_seal,
        interactions=args.interactions,
        random_state=args.seed
    )
    print(f"拟合 SEAL 输出的 Pearson: {reg_metrics['pearson']:.4f}")
    print(f"拟合 SEAL 输出的 MSE: {reg_metrics['mse']:.6f}")

    # 对 SEAL-EBM 也计算 “用可解释层替代 SEAL 时，对 label 的 AUC/AP”
    y_seal_pred = ebm_reg.predict(X)
    auc_seal_ebm = roc_auc_score(y_true, y_seal_pred)
    ap_seal_ebm = average_precision_score(y_true, y_seal_pred)
    reg_metrics["auc_vs_label"] = float(auc_seal_ebm)
    reg_metrics["ap_vs_label"] = float(ap_seal_ebm)
    print(f"SEAL-EBM 替代 SEAL 对真实标签的 AUC: {auc_seal_ebm:.4f}")
    print(f"SEAL-EBM 替代 SEAL 对真实标签的 AP:  {ap_seal_ebm:.4f}")

    # 导出回归模型的全局解释（重要性 + shape）
    reg_global = ebm_reg.explain_global(name=f"EBM_SEAL_{dataset_name}_{args.split}")
    reg_imp_path = f"{dataset_name}_ebm_estimate_seal_global_{args.split}.npz"
    export_ebm_global(reg_global, FEATURE_NAMES, reg_imp_path)

    print("\n=== 训练 EBM_estimate_label（分类：拟合真实标签） ===")
    ebm_clf, clf_metrics = train_ebm_estimate_label(
        X, y_true,
        interactions=args.interactions,
        random_state=args.seed
    )
    print(f"分类 EBM 对真实标签的 AUC: {clf_metrics['auc']:.4f}")
    print(f"分类 EBM 对真实标签的 AP:  {clf_metrics['ap']:.4f}")

    # 导出分类模型的全局解释（重要性 + shape）
    clf_global = ebm_clf.explain_global(name=f"EBM_LABEL_{dataset_name}_{args.split}")
    clf_imp_path = f"{dataset_name}_ebm_estimate_label_global_{args.split}.npz"
    export_ebm_global(clf_global, FEATURE_NAMES, clf_imp_path)

    # 保存样本级预测，用于后续统计 / case study
    out_samples_path = f"{dataset_name}_ebm_samples_{args.split}.npz"
    proba_label = ebm_clf.predict_proba(X)[:, 1]
    np.savez(
        out_samples_path,
        X=X.astype(np.float32),
        y_true=y_true.astype(np.float32),
        y_seal=y_seal.astype(np.float32),
        y_seal_pred=y_seal_pred.astype(np.float32),
        y_ebm_label_proba=proba_label.astype(np.float32),
        reg_metrics=reg_metrics,
        clf_metrics=clf_metrics,
    )
    print(f"\n样本级特征与预测已保存到: {out_samples_path}")

    # 可选：保存 local explanation（体积巨大，不建议对所有样本启用）
    if args.save_local:
        print("\n=== 计算并保存 local explanations（注意：可能非常耗时和占空间） ===")
        # 对 SEAL-EBM
        reg_local = ebm_reg.explain_local(X, y_seal, name=f"EBM_SEAL_LOCAL_{dataset_name}_{args.split}")
        reg_local_data = reg_local.data()  # dict with 'specific', 'overall' 等字段
        np.savez(f"{dataset_name}_ebm_seal_local_{args.split}.npz", **reg_local_data)
        # 对 Label-EBM
        clf_local = ebm_clf.explain_local(X, y_true, name=f"EBM_LABEL_LOCAL_{dataset_name}_{args.split}")
        clf_local_data = clf_local.data()
        np.savez(f"{dataset_name}_ebm_label_local_{args.split}.npz", **clf_local_data)
        print("local explanations 已保存。")

    print("\n完成。你现在有：")
    print("  - *_ebm_estimate_seal_global_*.npz ：SEAL-EBM 的 importance + shape")
    print("  - *_ebm_estimate_label_global_*.npz ：Label-EBM 的 importance + shape")
    print("  - *_ebm_samples_*.npz ：样本级特征与 EBM / SEAL 预测，用于 AUC/AP 统计和 case study")
    print("  - (可选) *_ebm_*_local_*.npz ：局部贡献，用于详细 case study")

if __name__ == "__main__":
    main()
