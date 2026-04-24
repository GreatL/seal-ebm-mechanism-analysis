#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from interpret.glassbox import ExplainableBoostingClassifier

#############################################
# 你需要把 compute_edge_features 替换成你实际的特征提取函数
#############################################

FEATURE_LIST = [
    "deg_u","deg_v","deg_min","deg_max","deg_sum","deg_prod",
    "CN","AA","RA","Jaccard","sp_len","tri_u","tri_v","tri_uv",
]

def compute_edge_features(G, u, v):
    """
    简化版结构特征；你可以按你的论文特征集自己完善。
    这里只给一个示意，保证脚本可跑。
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
        sp_len = 10  # 也可以用直径上界或其他

    # 局部 triangle 数量（简单方法）
    tri_u = sum(1 for (x, y) in nx.generators.classic.complete_graph(list(Nu)).edges()
                if G.has_edge(x, y)) if len(Nu) >= 2 else 0
    tri_v = sum(1 for (x, y) in nx.generators.classic.complete_graph(list(Nv)).edges()
                if G.has_edge(x, y)) if len(Nv) >= 2 else 0
    tri_uv = cn  # 封闭三角形和 CN 在无向图上等价，这里做个 proxy

    return np.array([
        deg_u, deg_v, deg_min, deg_max, deg_sum, deg_prod,
        cn, aa, ra, jacc, sp_len, tri_u, tri_v, tri_uv
    ], dtype=float)

#############################################

def degree_preserving_randomization(G, n_swaps_factor=10, seed=0):
    G_rand = G.copy()
    M = G_rand.number_of_edges()
    n_swaps = n_swaps_factor * M
    rng = np.random.RandomState(seed)
    # networkx>=2.8 支持 seed 参数
    nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps*10, seed=rng)
    return G_rand

def sample_edges_and_features(G, n_pos=5000, n_neg=5000, seed=0):
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())

    edges = list(G.edges())
    if len(edges) == 0:
        raise ValueError("Graph has no edges.")
    pos_idx = rng.choice(len(edges), size=min(n_pos, len(edges)), replace=False)
    pos_edges = [edges[i] for i in pos_idx]

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
    p = ebm.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)

    exp = ebm.explain_global(name="LB-EBM")
    data = exp.data()
    # interpret 会按顺序命名 feature_0000, feature_0001,...
    names = data["names"]
    scores = data["scores"]
    name2score = {n: float(s) for n, s in zip(names, scores)}

    imp = []
    for j in range(len(FEATURE_LIST)):
        internal = f"feature_{j:04d}"
        imp.append(name2score.get(internal, 0.0))
    imp = np.array(imp, dtype=float)
    return imp, auc, ap

def main():
    ds = "FBK"   # 换成你想做随机化实验的网络名
    path = f"data/{ds}.txt"  # 改成你的真实路径
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)
    print(f"[INFO] {ds}: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    print("[INFO] sampling edges & training Label-EBM on ORIGINAL graph...")
    X_orig, y_orig = sample_edges_and_features(G, n_pos=5000, n_neg=5000, seed=0)
    imp_orig, auc_orig, ap_orig = train_label_ebm(X_orig, y_orig, seed=0)
    print(f"[ORIG] AUC={auc_orig:.4f}, AP={ap_orig:.4f}")
    print("[ORIG] feature importances:")
    for f, v in zip(FEATURE_LIST, imp_orig):
        print(f"  {f:10s}: {v:.4f}")

    print("\n[INFO] degree-preserving randomization...")
    G_rand = degree_preserving_randomization(G, n_swaps_factor=10, seed=1)

    print("[INFO] sampling edges & training Label-EBM on RANDOMIZED graph...")
    X_rand, y_rand = sample_edges_and_features(G_rand, n_pos=5000, n_neg=5000, seed=1)
    imp_rand, auc_rand, ap_rand = train_label_ebm(X_rand, y_rand, seed=1)
    print(f"[RAND] AUC={auc_rand:.4f}, AP={ap_rand:.4f}")
    print("[RAND] feature importances:")
    for f, v in zip(FEATURE_LIST, imp_rand):
        print(f"  {f:10s}: {v:.4f}")

    print("\n[COMPARE] orig vs randomized (key features):")
    key_feats = ["tri_uv", "Jaccard", "RA", "sp_len", "deg_u", "deg_prod"]
    for f in key_feats:
        if f not in FEATURE_LIST:
            continue
        j = FEATURE_LIST.index(f)
        print(f"{f:10s}: orig={imp_orig[j]:.4f}, rand={imp_rand[j]:.4f}")

if __name__ == "__main__":
    main()
