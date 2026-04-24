#!/usr/bin/env python3
import argparse
import os
import csv
import random
import warnings
import time
from datetime import datetime

import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.nn import GCNConv, global_sort_pool


warnings.filterwarnings("ignore", category=UserWarning)


###############################################################################
# Seed
###############################################################################
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# 参数统计
###############################################################################
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


###############################################################################
# Edge list
###############################################################################
def load_edge_list(path):
    raw_edges = []
    node_map = {}
    cur = 0

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = parts[:2]
            if u not in node_map:
                node_map[u] = cur
                cur += 1
            if v not in node_map:
                node_map[v] = cur
                cur += 1
            raw_edges.append((node_map[u], node_map[v]))

    raw_edges = np.array(raw_edges).T
    return raw_edges, len(node_map)


###############################################################################
# 划分
###############################################################################
def do_edge_split(edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    n = row.size(0)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    perm = torch.randperm(n)
    row, col = row[perm], col[perm]

    split = {"train": {}, "valid": {}, "test": {}}

    split["valid"]["pos"] = torch.stack([row[:n_val], col[:n_val]])
    split["test"]["pos"] = torch.stack([row[n_val:n_val+n_test],
                                        col[n_val:n_val+n_test]])
    split["train"]["pos"] = torch.stack([row[n_val+n_test:], col[n_val+n_test:]])

    neg = negative_sampling(edge_index=edge_index,
                            num_nodes=num_nodes,
                            num_neg_samples=n)

    split["valid"]["neg"] = neg[:, :n_val]
    split["test"]["neg"] = neg[:, n_val:n_val+n_test]
    split["train"]["neg"] = neg[:, n_val+n_test:]

    return split


###############################################################################
# K-hop
###############################################################################
def neighbors(fringe, A):
    return set(A[list(fringe)].indices)


def k_hop_subgraph(src, dst, k, A):
    nodes = [src, dst]
    visited = set([src, dst])
    fringe = set([src, dst])

    for _ in range(1, k + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited |= fringe
        if len(fringe) == 0:
            break
        nodes += list(fringe)

    sub = A[nodes][:, nodes]
    sub[0, 1] = 0
    sub[1, 0] = 0
    return nodes, sub


###############################################################################
# DRNL
###############################################################################
def drnl(adj, src, dst):
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_s = adj[idx][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_d = adj[idx][:, idx]

    d2s = shortest_path(adj_d, directed=False, unweighted=True, indices=src)
    d2d = shortest_path(adj_s, directed=False, unweighted=True, indices=dst - 1)

    d2s = np.insert(d2s, dst, 0)
    d2d = np.insert(d2d, src, 0)

    dist = d2s + d2d
    dist[np.isinf(dist)] = np.nan

    dist2 = dist // 2
    distm = dist % 2

    z = 1 + np.minimum(d2s, d2d)
    z += dist2 * (dist2 + distm - 1)

    z[src] = 1
    z[dst] = 1
    z[np.isnan(z)] = 0

    return torch.LongTensor(z)


###############################################################################
# 构造子图 Data
###############################################################################
def construct_graph(sub, y):
    u, v, _ = ssp.find(sub)
    edge_index = torch.LongTensor([u, v])
    z = drnl(sub, 0, 1)
    return Data(edge_index=edge_index, z=z,
                y=torch.tensor([y]), num_nodes=sub.shape[0])


def extract(edge_index, A, k, y):
    data_list = []
    for u, v in tqdm(edge_index.T.tolist(), disable=True):
        _, sub = k_hop_subgraph(u, v, k, A)
        data_list.append(construct_graph(sub, y))
    return data_list


###############################################################################
# DGCNN
###############################################################################
class DGCNN(torch.nn.Module):
    def __init__(self, hidden, layers, max_z, k):
        super().__init__()

        self.emb = Embedding(max_z, hidden)

        self.convs = ModuleList()
        self.convs.append(GCNConv(hidden, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, 1))

        self.k = k
        total = hidden * layers + 1

        self.conv1 = Conv1d(1, 16, total, total)
        self.pool = MaxPool1d(2, 2)
        self.conv2 = Conv1d(16, 32, 5, 1)

        dense = int((k - 2) / 2 + 1)
        dense = (dense - 5 + 1) * 32

        self.lin1 = Linear(dense, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch):
        x = self.emb(z)

        xs = [x]
        for c in self.convs:
            xs += [torch.tanh(c(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


###############################################################################
# Train & Test
###############################################################################
def train(model, loader, opt, device):
    model.train()
    total = 0
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        logit = model(d.z, d.edge_index, d.batch)
        loss = F.binary_cross_entropy_with_logits(logit.view(-1), d.y.float())
        loss.backward()
        opt.step()
        total += loss.item() * d.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    yp, yt = [], []

    for d in loader:
        d = d.to(device)
        logit = model(d.z, d.edge_index, d.batch)
        yp.append(logit.view(-1).cpu())
        yt.append(d.y.cpu())

    yp = torch.cat(yp).numpy()
    yt = torch.cat(yt).numpy()

    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
    return auc, ap, peak


###############################################################################
# Utils
###############################################################################
def get_dataset_name(edge_path):
    return os.path.splitext(os.path.basename(edge_path))[0]


def ratio_str(x):
    return str(round(float(x), 4)).rstrip("0").rstrip(".") if "." in str(round(float(x),4)) else str(x)


def get_cache_path(args):
    os.makedirs("cache", exist_ok=True)
    dataset = get_dataset_name(args.edge_path)
    fname = (
        f"{dataset}"
        f"__h{args.num_hops}"
        f"__val{ratio_str(args.val_ratio)}"
        f"__test{ratio_str(args.test_ratio)}"
        f"__seed{args.seed}.pt"
    )
    return os.path.join("cache", fname)


def get_model_path(args):
    os.makedirs(args.model_dir, exist_ok=True)
    name = args.model_name or get_dataset_name(args.edge_path)
    return os.path.join(args.model_dir, f"{name}_seal.pt")


def save_result_csv(args, n_nodes,
                    mean_auc, std_auc,
                    mean_ap, std_ap,
                    total_params, trainable_params,
                    mean_forward, std_forward,
                    mean_total_mem, std_total_mem,
                    mean_train, std_train,
                    mean_total_time, std_total_time):

    dataset = get_dataset_name(args.edge_path)
    csv_path = f"{dataset}.csv"
    new_file = not os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if new_file:
            w.writerow([
                "timestamp","dataset","mode","model_name",
                "nodes","runs","seed",
                "num_hops","epochs","batch_size",
                "hidden","layers",
                "val_ratio","test_ratio",
                "Params_total","Params_trainable",
                "AUC_mean","AUC_std","AP_mean","AP_std",
                "ForwardMem_meanMB","ForwardMem_stdMB",
                "TotalMem_meanMB","TotalMem_stdMB",
                "TrainTime_mean","TrainTime_std",
                "TotalTime_mean","TotalTime_std"
            ])

        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset,
            args.mode,
            args.model_name,
            n_nodes,
            args.runs,
            args.seed,
            args.num_hops,
            args.epochs,
            args.batch_size,
            args.hidden,
            args.layers,
            args.val_ratio,
            args.test_ratio,
            total_params,
            trainable_params,
            float(mean_auc),
            float(std_auc),
            float(mean_ap),
            float(std_ap),
            float(mean_forward) if mean_forward is not None else None,
            float(std_forward)  if std_forward  is not None else None,
            float(mean_total_mem) if mean_total_mem is not None else None,
            float(std_total_mem)  if std_total_mem  is not None else None,
            float(mean_train) if mean_train is not None else None,
            float(std_train)  if std_train  is not None else None,
            float(mean_total_time),
            float(std_total_time)
        ])

    print(f"结果写入: {csv_path}")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train","test"])
    parser.add_argument("--edge_path", required=True)

    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)

    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=1)

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--model_name", default=None)

    parser.add_argument("--log_dir", default="logs")
    # 严格训练图模式：训练子图只基于训练正边构造
    parser.add_argument("--strict_train_graph", action="store_true")
    # 在 strict_train_graph 模式下，验证/测试子图是否用完整图
    # 默认为 False，即 val/test 也只在训练图上抽子图（最保守）
    parser.add_argument("--eval_use_full_graph", action="store_true")
    
    args = parser.parse_args()

    all_auc, all_ap = [], []
    all_train_time, all_total_time = [], []
    all_forward_peak, all_total_peak = [], []

    total_params = None
    trainable_params = None

    for r in range(args.runs):

        print(f"\n===== Run {r+1}/{args.runs} =====\n")
        setup_seed(args.seed + r)

        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        run_start = time.time()

        cache_path = get_cache_path(args)
        use_cache = False

        ###################################################################
        # 尝试读取缓存
        ###################################################################
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu")
            if (
                cache.get("edge_path") == args.edge_path
                and cache.get("num_hops") == args.num_hops
                and abs(cache.get("val_ratio") - args.val_ratio) < 1e-12
                and abs(cache.get("test_ratio") - args.test_ratio) < 1e-12
                and cache.get("seed") == args.seed
            ):
                print(f"已载入缓存数据集: {cache_path}")
                train_data = cache["train_data"]
                val_data = cache["val_data"]
                test_data = cache["test_data"]
                n = cache["num_nodes"]
                use_cache = True
            else:
                print("检测到缓存参数不一致，忽略缓存")
                print(cache.get("edge_path"),cache.get("num_hops"),cache.get("val_ratio"),cache.get("test_ratio"),cache.get("seed"))
                print(args.edge_path,args.num_hops,args.val_ratio,args.test_ratio,args.seed)

        ##################################################################  
        # 若无缓存 → 重新构建
        ###################################################################
        if not use_cache:

            e, n = load_edge_list(args.edge_path)
            edge_index = torch.LongTensor(e)
            edge_index = to_undirected(edge_index)
            # 先做边划分
            split = do_edge_split(edge_index, n, args.val_ratio, args.test_ratio)
            # ===== 构造邻接矩阵 =====
            # A_full：完整图，包含所有正边（train+valid+test）
            A_full = ssp.csr_matrix(
                (np.ones(edge_index.size(1)),
                (edge_index[0], edge_index[1])),
                shape=(n, n)
            )

            # A_train：只包含训练正边
            train_pos = split["train"]["pos"]  # shape [2, E_train]
            A_train = ssp.csr_matrix(
                (np.ones(train_pos.size(1)),
                (train_pos[0], train_pos[1])),
                shape=(n, n)
            )
            # 因为 do_edge_split 只保留 row<col，因此需要补齐对称
            A_train = A_train + A_train.T

            # 根据配置决定各阶段使用的邻接矩阵
            if args.strict_train_graph:
                # 训练：只用训练图
                A_train_sub = A_train
                # 验证/测试：默认也用训练图；若 eval_use_full_graph=True，则用完整图
                if args.eval_use_full_graph:
                    A_eval_sub = A_full
                else:
                    A_eval_sub = A_train
            else:
                # 非严格模式：所有阶段都用完整图（与原始 SEAL 行为相同）
                A_train_sub = A_full
                A_eval_sub = A_full

            print("Extracting train subgraphs...")
            train_data = extract(split["train"]["pos"], A_train_sub, args.num_hops, 1) + \
                        extract(split["train"]["neg"], A_train_sub, args.num_hops, 0)

            print("Extracting valid subgraphs...")
            val_data = extract(split["valid"]["pos"], A_eval_sub, args.num_hops, 1) + \
                    extract(split["valid"]["neg"], A_eval_sub, args.num_hops, 0)

            print("Extracting test subgraphs...")
            test_data = extract(split["test"]["pos"], A_eval_sub, args.num_hops, 1) + \
                        extract(split["test"]["neg"], A_eval_sub, args.num_hops, 0)

            torch.save({
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "num_nodes": n,
                "edge_path": args.edge_path,
                "num_hops": args.num_hops,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed
            }, cache_path)

            print(f"数据缓存至: {cache_path}")

        ###################################################################
        # DataLoader
        ###################################################################
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DGCNN(args.hidden, args.layers, 1000, 30).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        if total_params is None:
            total_params, trainable_params = count_parameters(model)
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")

        model_path = get_model_path(args)

        ###################################################################
        # TEST ONLY
        ###################################################################
        if args.mode == "test":

            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"已加载模型: {model_path}")

            test_auc, test_ap, forward_peak = test(model, test_loader, device)
            total_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
            total_time = time.time() - run_start

            print(f"TEST AUC = {test_auc:.4f}")
            print(f"TEST AP  = {test_ap:.4f}")
            print(f"Forward Peak Memory = {forward_peak} MB")
            print(f"Total Peak Memory   = {total_mem} MB")
            print(f"测试总耗时: {total_time:.2f} 秒")

            all_auc.append(test_auc)
            all_ap.append(test_ap)
            all_forward_peak.append(forward_peak)
            all_total_peak.append(total_mem)
            all_total_time.append(total_time)

            continue

        ###################################################################
        # TRAIN
        ###################################################################
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = open(os.path.join(
            args.log_dir,
            f"{args.model_name or get_dataset_name(args.edge_path)}_log.csv"
        ), "w", newline="", encoding="utf-8")

        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch","loss","val_auc","val_ap"])

        train_start = time.time()

        for ep in range(1, args.epochs + 1):
            loss = train(model, train_loader, opt, device)
            val_auc, val_ap, _ = test(model, val_loader, device)
            print(f"Epoch {ep} Loss={loss:.4f} Val AUC={val_auc:.4f} Val AP={val_ap:.4f}")
            log_writer.writerow([ep, loss, val_auc, val_ap])
            log_file.flush()

        log_file.close()

        train_time = time.time() - train_start
        print(f"本次训练耗时: {train_time:.2f} 秒")

        test_auc, test_ap, forward_peak = test(model, test_loader, device)
        total_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
        total_time = time.time() - run_start

        print(f"FINAL TEST AUC = {test_auc:.4f}")
        print(f"FINAL TEST AP  = {test_ap:.4f}")
        print(f"Forward Peak Memory = {forward_peak} MB")
        print(f"Total Peak Memory   = {total_mem} MB")
        print(f"总耗时: {total_time:.2f} 秒")

        if args.save_model:
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "num_nodes": n
            }, model_path)
            print(f"模型保存至: {model_path}")

        all_auc.append(test_auc)
        all_ap.append(test_ap)
        all_train_time.append(train_time)
        all_total_time.append(total_time)
        all_forward_peak.append(forward_peak)
        all_total_peak.append(total_mem)

    ###################################################################
    # SUMMARY
    ###################################################################
    mean_auc = np.mean(all_auc)
    std_auc  = np.std(all_auc)
    mean_ap  = np.mean(all_ap)
    std_ap   = np.std(all_ap)

    mean_train = np.mean(all_train_time) if len(all_train_time) else None
    std_train  = np.std(all_train_time) if len(all_train_time) else None

    mean_total = np.mean(all_total_time)
    std_total  = np.std(all_total_time)

    mean_forward = np.mean(all_forward_peak) if any(all_forward_peak) else None
    std_forward  = np.std(all_forward_peak) if any(all_forward_peak) else None

    mean_total_mem = np.mean(all_total_peak) if any(all_total_peak) else None
    std_total_mem  = np.std(all_total_peak) if any(all_total_peak) else None

    print("\n===== SUMMARY =====")
    print(f"AUC mean={mean_auc:.4f} std={std_auc:.4f}")
    print(f"AP  mean={mean_ap:.4f} std={std_ap:.4f}")
    if mean_train is not None:
        print(f"TrainTime mean={mean_train:.2f}s std={std_train:.2f}s")
    print(f"TotalTime mean={mean_total:.2f}s std={std_total:.2f}s")
    if mean_forward is not None:
        print(f"Forward Peak Mem mean={mean_forward:.2f}MB std={std_forward:.2f}MB")
        print(f"Total Peak Mem   mean={mean_total_mem:.2f}MB std={std_total_mem:.2f}MB")
    print("===================\n")

    save_result_csv(
        args, n,
        mean_auc, std_auc,
        mean_ap, std_ap,
        total_params, trainable_params,
        mean_forward, std_forward,
        mean_total_mem, std_total_mem,
        mean_train, std_train,
        mean_total, std_total
    )
