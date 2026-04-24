#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
解释 SEAL（DGCNN）链路预测模型的树模型脚本：
1）加载缓存子图和训练好的 SEAL 模型；
2）为每个子图计算结构特征 φ(G_uv)；
3）获取 SEAL 对每个子图的输出概率；
4）训练决策树回归模型拟合 SEAL 输出；
5）输出树的规则和特征重要性。

依赖：
- 与 seal_from_edgelist.py 相同的环境
- scikit-learn, networkx
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as ssp
import networkx as nx

from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

###########################################################
# 复制/复用 seal_from_edgelist.py 中的一些定义
###########################################################

from seal_from_edgelist import (
    DGCNN,
    get_cache_path,
    get_model_path,
    get_dataset_name,
)


def compute_struct_features(sub_adj: ssp.csr_matrix,
                            src: int = 0,
                            dst: int = 1):
    """
    给定子图邻接矩阵 sub_adj (csr)，其中 src 和 dst 是中心端点在子图中的索引
    （在你的构造中就是 0 和 1），计算一组可解释的结构特征。

    返回: 一维 numpy array, 形状 (d,)
    """
    # 构建 networkx 图
    # sub_adj 是无向稀疏矩阵
    G = nx.from_scipy_sparse_array(sub_adj)

    n_sub = G.number_of_nodes()
    m_sub = G.number_of_edges()

    # ---- 子图规模与基础统计 ----
    # 节点数、边数、平均度
    avg_deg = 2 * m_sub / n_sub if n_sub > 0 else 0.0

    # 子图平均聚类系数（对较大子图可能开销略高，必要时可关掉）
    try:
        avg_clust = nx.average_clustering(G)
    except Exception:
        avg_clust = 0.0

    # ---- 中心端点的度及组合 ----
    deg_u = G.degree[src] if src in G else 0
    deg_v = G.degree[dst] if dst in G else 0

    deg_min = min(deg_u, deg_v)
    deg_max = max(deg_u, deg_v)
    deg_sum = deg_u + deg_v
    deg_prod = deg_u * deg_v  # Preferential Attachment

    # ---- 公共邻居与经典启发式 ----
    # 公共邻居集合
    if src in G and dst in G:
        cn_nodes = list(nx.common_neighbors(G, src, dst))
    else:
        cn_nodes = []

    cn = len(cn_nodes)

    # Adamic-Adar, Resource Allocation, Jaccard
    aa = 0.0
    ra = 0.0
    for w in cn_nodes:
        dw = G.degree[w]
        if dw > 1:
            aa += 1.0 / np.log(dw + 1e-8)
            ra += 1.0 / dw

    # Jaccard
    try:
        neigh_u = set(G.neighbors(src)) if src in G else set()
        neigh_v = set(G.neighbors(dst)) if dst in G else set()
        inter = len(neigh_u & neigh_v)
        union = len(neigh_u | neigh_v)
        jaccard = inter / union if union > 0 else 0.0
    except Exception:
        jaccard = 0.0

    # ---- u,v 的最短路径长度（注意 SEAL 构造时已经把 (u,v) 边删掉）----
    try:
        sp_len = nx.shortest_path_length(G, source=src, target=dst)
    except nx.NetworkXNoPath:
        sp_len = 0  # 或者设为一个特殊值，比如 -1 或 n_sub+1

    # ---- 简单 motif：以 u 为端点的三角形个数、以 v 为端点的三角形个数 ----
    # networkx.triangles 返回每个节点参与的三角形数量
    try:
        triangles = nx.triangles(G)
        tri_u = triangles.get(src, 0)
        tri_v = triangles.get(dst, 0)
        # 包含 u,v 的三角形个数：公共邻居节点 w 使得 u-w-v 形成三角形
        tri_uv = 0
        for w in cn_nodes:
            if G.has_edge(src, w) and G.has_edge(dst, w):
                tri_uv += 1
    except Exception:
        tri_u = tri_v = tri_uv = 0

    # ---- 可以根据需要继续扩展 motif/path 特征 ----

    # 汇总成特征向量
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


def collect_features_and_seal_outputs(cache_path,
                                      model_path,
                                      device,
                                      use_split="test",
                                      batch_size=64,
                                      max_samples=None):
    """
    从缓存和模型中：
    1）加载 train/val/test 子图；
    2）对指定 split (train/val/test) 计算结构特征和 SEAL 输出。

    返回:
      X: np.ndarray, shape [N, d]  结构特征
      y_seal: np.ndarray, [N]     SEAL 概率输出
      y_true: np.ndarray, [N]     真实标签
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

    # 加载模型
    ckpt = torch.load(model_path, map_location=device)
    model_args = ckpt.get("args", None)
    num_nodes = ckpt.get("num_nodes", None)

    # 如果模型构造参数固定在主脚本，这里手动对齐
    # 原脚本中：model = DGCNN(args.hidden, args.layers, 1000, 30)
    # 为简单起见，假设你一直用同样的 hidden/layers/max_z/k
    hidden = model_args["hidden"] if model_args is not None else 32
    layers = model_args["layers"] if model_args is not None else 3

    # 这里的 max_z, k 需要与训练时一致，若你在主脚本中改过，请保持一致
    max_z = 1000
    k = 30

    model = DGCNN(hidden, layers, max_z, k).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 准备 DataLoader 得到 batch.batch 等
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_feats = []
    all_y_seal = []
    all_y_true = []

    # 遍历 batch，获取模型输出 + 结构特征
    with torch.no_grad():
        for i, d in enumerate(loader):
            d = d.to(device)
            # DGCNN forward
            logit = model(d.z, d.edge_index, d.batch)  # [batch_size, 1]
            prob = torch.sigmoid(logit.view(-1)).cpu().numpy()
            labels = d.y.view(-1).cpu().numpy()

            # 逐个子图计算结构特征
            # 注意 Data 对象中没有直接存子图邻接矩阵，只给 edge_index
            # 这里我们从 edge_index 构建临时邻接矩阵
            # 但要注意：子图节点已经是 re-indexed 0..num_nodes_sub-1
            edge_index = d.edge_index.cpu().numpy()
            batch_vec = d.batch.cpu().numpy()
            num_graphs = d.num_graphs

            # 对每个图分别提取子 edge_index
            for g_idx in range(num_graphs):
                mask = (batch_vec == g_idx)
                # 子图的节点在此 batch 中的局部索引
                nodes_in_g = np.where(mask)[0]
                # 建立局部映射
                local_id = {nid: i for i, nid in enumerate(nodes_in_g)}
                # 筛选属于该图的边
                e_mask = np.isin(edge_index[0], nodes_in_g)
                e_idx = edge_index[:, e_mask]

                # 重新映射到 0..n_sub-1
                #u = np.vectorize(local_id.get)(e_idx[0])
                #v = np.vectorize(local_id.get)(e_idx[1])
                if e_idx[0].size == 0 or e_idx[1].size == 0:
                    u = np.array([], dtype=int)
                    v = np.array([], dtype=int)
                else:
                    u = np.array([local_id.get(x, -1) for x in e_idx[0]], dtype=int)
                    v = np.array([local_id.get(x, -1) for x in e_idx[1]], dtype=int)
                    # 如果不希望出现缺失节点，可以 assert -1 不存在
                    assert (u != -1).all() and (v != -1).all()

                n_sub = len(nodes_in_g)
                sub_adj = ssp.csr_matrix(
                    (np.ones(len(u), dtype=np.float32), (u, v)),
                    shape=(n_sub, n_sub)
                )
                # 由于原子图是无向的，这里对称化一下
                sub_adj = sub_adj + sub_adj.T
                sub_adj.data[:] = 1.0
                sub_adj.eliminate_zeros()

                # 按 SEAL 构造约定，中心端点应对应子图中的 0 和 1
                # 在主脚本中，k_hop_subgraph 是先 nodes=[src,dst] 再扩展，
                # 因此 src,dst 在子图中的索引就是 0,1。
                feats = compute_struct_features(sub_adj, src=0, dst=1)

                all_feats.append(feats)

            all_y_seal.extend(prob.tolist())
            all_y_true.extend(labels.tolist())

            # 如果想限制样本数量以加速，可以用 max_samples
            if max_samples is not None and len(all_feats) >= max_samples:
                break

    X = np.vstack(all_feats)
    y_seal = np.array(all_y_seal[:len(all_feats)], dtype=np.float32)
    y_true = np.array(all_y_true[:len(all_feats)], dtype=np.float32)

    return X, y_seal, y_true


def train_tree_model(X, y_seal, max_depth=4, min_samples_leaf=20, random_state=0):
    """
    使用决策树回归模型拟合 SEAL 输出。
    """
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    tree.fit(X, y_seal)
    return tree


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--edge_path", required=True,
                        help="与 seal_from_edgelist.py 使用的 edge_path 一致")
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--model_name", default=None)

    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="在哪个 split 上做解释/拟合")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多使用多少个子图样本来训练树，None 表示全部")

    parser.add_argument("--tree_max_depth", type=int, default=4)
    parser.add_argument("--tree_min_samples_leaf", type=int, default=20)

    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 对齐 cache_path 和 model_path 的逻辑
    class DummyArgs:
        # 用于复用 get_cache_path 和 get_model_path
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
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}\n"
                                f"请先运行 seal_from_edgelist.py 生成缓存。")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n"
                                f"请先用 seal_from_edgelist.py --save_model 训练并保存模型。")

    print(f"使用缓存: {cache_path}")
    print(f"使用模型: {model_path}")

    # 收集特征和 SEAL 输出
    print(f"收集 {args.split} 子图的结构特征和 SEAL 输出...")
    X, y_seal, y_true = collect_features_and_seal_outputs(
        cache_path,
        model_path,
        device,
        use_split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    # 训练决策树回归模型
    print("训练决策树回归模型拟合 SEAL 输出...")
    tree = train_tree_model(
        X, y_seal,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.seed
    )

    # 评估拟合质量
    y_pred = tree.predict(X)
    mse = mean_squared_error(y_seal, y_pred)
    corr, _ = pearsonr(y_seal, y_pred)

    print(f"拟合 SEAL 输出的 MSE: {mse:.6f}")
    print(f"拟合 SEAL 输出的 Pearson 相关: {corr:.4f}")

    # 输出特征重要性
    feature_names = [
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
    importances = tree.feature_importances_
    print("\n特征重要性（按从高到低排序）:")
    idx_sorted = np.argsort(importances)[::-1]
    for idx in idx_sorted:
        if importances[idx] <= 0:
            continue
        print(f"  {feature_names[idx]:>10s}: {importances[idx]:.4f}")

    # 导出树的规则
    dataset_name = get_dataset_name(args.edge_path)
    tree_rule_path = f"{dataset_name}_seal_tree_rules.txt"
    rules = export_text(tree, feature_names=feature_names)
    with open(tree_rule_path, "w", encoding="utf-8") as f:
        f.write(rules)

    print(f"\n决策树规则已写入: {tree_rule_path}")
    print("你可以直接阅读该文件，获得类似 if-then 的解析规则，")
    print("从中观察 SEAL 在当前网络上的近似结构聚合模式。")


if __name__ == "__main__":
    main()
