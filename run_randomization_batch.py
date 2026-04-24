#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from interpret.glassbox import ExplainableBoostingClassifier

###########################################################
# 配置部分
###########################################################

# 要跑随机化实验的网络名（按你的 data/<name>.txt 命名）
DATASETS = [
    "NSC",      # 你已经跑过的
    "FBK",
    "USAir",
    "Power",
    "Yeast",
    # 按需添加/删除
]

# 对应的边文件路径模式
EDGE_PATH_TEMPLATE = "data/{ds}.txt"   # 如果不在 data/ 目录里，这里改成你的真实路径模板

# 特征列表（顺序必须和 compute_edge_features 返回顺序一致）
FEATURE_LIST = [
    "deg_u","deg_v","deg_min","deg_max","deg_sum","deg_prod",
    "CN","AA","RA","Jaccard","sp_len","tri_u","tri_v","tri_uv",
]

# 重点画图关注的特征
KEY_FEATS = ["tri_uv", "Jaccard", "RA", "CN", "sp_len", "deg_u", "deg_prod"]

###########################################################
# 你可以在这里替换成你 SEAL 项目里真实的特征函数
###########################################################

def compute_edge_features(G, u, v):
    """
    简化版结构特征；建议用你项目里已有的特征函数替换。
    """
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    deg_min = min(deg_u, deg_v)
    deg_max = max(deg_u, deg_v)
    deg_sum = deg_u + deg_v
    deg_prod = deg_u * deg_v

    # 邻居集合
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    cn = len(Nu & Nv)
    aa = sum(1.0 / np.log(G.degree(w)+1e-6) for w in (Nu & Nv)) if cn > 0 else 0.0
    ra = sum(1.0 / G.degree(w) for w in (Nu & Nv)) if cn > 0 else 0.0
    jacc = cn / len(Nu | Nv) if len(Nu | Nv) > 0 else 0.0

    # 最短路径长度（断开的话给一个较大值）
    try:
        sp_len = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        sp_len = 10  # 可按直径上界调整

    # 局部三角形示意（这里给的是粗略版，建议用你真实的 tri_u/tri_v/tri_uv 定义替换）
    tri_u = 0
    if len(Nu) >= 2:
        for x in Nu:
            for y in Nu:
                if x < y and G.has_edge(x, y):
                    tri_u += 1
    tri_v = 0
    if len(Nv) >= 2:
        for x in Nv:
            for y in Nv:
                if x < y and G.has_edge(x, y):
                    tri_v += 1
    tri_uv = cn  # 在简化版里用 CN 作为 proxy

    return np.array([
        deg_u, deg_v, deg_min, deg_max, deg_sum, deg_prod,
        cn, aa, ra, jacc, sp_len, tri_u, tri_v, tri_uv
    ], dtype=float)

###########################################################
# 随机化 / 采样 / EBM 训练
###########################################################

def degree_preserving_randomization(G, n_swaps_factor=10, seed=0):
    G_rand = G.copy()
    M = G_rand.number_of_edges()
    n_swaps = n_swaps_factor * M
    rng = np.random.RandomState(seed)
    nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 10, seed=rng)
    return G_rand

def sample_edges_and_features(G, n_pos=5000, n_neg=5000, seed=0):
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    edges = list(G.edges())
    if len(edges) == 0:
        raise ValueError("Graph has no edges.")

    # 正例：从现有边均匀采样
    n_pos_use = min(n_pos, len(edges))
    pos_idx = rng.choice(len(edges), size=n_pos_use, replace=False)
    pos_edges = [edges[i] for i in pos_idx]

    # 负例：随机采未连边
    neg_edges = set()
    while len(neg_edges) < n_neg:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        if G.has_edge(u, v):
            continue
        if (u, v) in neg_edges or (v, u) in neg_edges:
            continue
        neg_edges.add((u, v))
    neg_edges = list(neg_edges)

    X_list = []
    y_list = []
    for (u, v) in pos_edges:
        X_list.append(compute_edge_features(G, u, v))
        y_list.append(1)
    for (u, v) in neg_edges:
        X_list.append(compute_edge_features(G, u, v))
        y_list.append(0)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y

def train_label_ebm(X, y, seed=0):
    ebm = ExplainableBoostingClassifier(
        interactions=0, max_bins=256, learning_rate=0.01, random_state=seed
    )
    ebm.fit(X, y)
    p = ebm.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)

    exp = ebm.explain_global(name="LB-EBM")
    data = exp.data()
    names = data["names"]
    scores = data["scores"]
    name2score = {n: float(s) for n, s in zip(names, scores)}

    imp = []
    for j in range(len(FEATURE_LIST)):
        internal = f"feature_{j:04d}"
        imp.append(name2score.get(internal, 0.0))
    imp = np.array(imp, dtype=float)
    return imp, auc, ap

###########################################################
# 主流程：遍历多个数据集，保存结果
###########################################################

def run_for_dataset(ds, edge_path):
    if not os.path.exists(edge_path):
        print(f"[WARN] skip {ds}: {edge_path} not found")
        return None

    print(f"\n[DATASET] {ds}")
    G = nx.read_edgelist(edge_path, nodetype=int)
    G = nx.Graph(G)
    print(f"[INFO] |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 原图
    print("[INFO] training Label-EBM on ORIGINAL graph...")
    X_orig, y_orig = sample_edges_and_features(G, n_pos=5000, n_neg=5000, seed=0)
    imp_orig, auc_orig, ap_orig = train_label_ebm(X_orig, y_orig, seed=0)
    print(f"[ORIG] AUC={auc_orig:.4f}, AP={ap_orig:.4f}")

    # 随机化图
    print("[INFO] degree-preserving randomization...")
    G_rand = degree_preserving_randomization(G, n_swaps_factor=10, seed=1)

    print("[INFO] training Label-EBM on RANDOMIZED graph...")
    X_rand, y_rand = sample_edges_and_features(G_rand, n_pos=5000, n_neg=5000, seed=1)
    imp_rand, auc_rand, ap_rand = train_label_ebm(X_rand, y_rand, seed=1)
    print(f"[RAND] AUC={auc_rand:.4f}, AP={ap_rand:.4f}")

    # 整理成 DataFrame 行
    rows = []
    for j, feat in enumerate(FEATURE_LIST):
        rows.append({
            "Dataset": ds,
            "Feature": feat,
            "Importance_orig": imp_orig[j],
            "Importance_rand": imp_rand[j],
            "AUC_orig": auc_orig,
            "AP_orig": ap_orig,
            "AUC_rand": auc_rand,
            "AP_rand": ap_rand,
        })
    return pd.DataFrame(rows)

def main():
    all_rows = []
    for ds in DATASETS:
        path = EDGE_PATH_TEMPLATE.format(ds=ds)
        df_ds = run_for_dataset(ds, path)
        if df_ds is not None:
            all_rows.append(df_ds)

    if not all_rows:
        print("[ERROR] no dataset successfully processed.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    out_path = "randomization_importances.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n[DONE] results saved to {out_path}")
    print(df_all.head())

if __name__ == "__main__":
    main()
