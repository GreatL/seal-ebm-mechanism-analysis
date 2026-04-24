import os
import glob
import numpy as np
import pandas as pd

# 和 explain_seal_gam.py 中保持一致
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


def parse_filename(path: str):
    """
    解析文件名，返回 (dataset, kind, split)

    适配你的命名：
      {dataset_name}_ebm_estimate_seal_importance_{split}.npz
      {dataset_name}_ebm_estimate_label_importance_{split}.npz

    例如：FBK_ebm_estimate_seal_importance_test.npz
      -> dataset='FBK', kind='seal', split='test'
    """
    fname = os.path.basename(path)
    if fname.endswith(".npz"):
        fname = fname[:-4]

    parts = fname.split("_")
    # 预期结构: [dataset, 'ebm', 'estimate', 'seal'/'label', 'importance', split]
    if len(parts) < 6:
        raise ValueError(f"文件名格式不符合预期: {os.path.basename(path)}")

    dataset = parts[0]
    kind = parts[3]   # 'seal' or 'label'
    split = parts[5]  # 'train'/'valid'/'test' 等

    return dataset, kind, split


def load_importance_npz(path: str) -> dict:
    """
    从一个 npz 文件中读取所有 importance 字段，返回:
        { feature_name: importance_value }

    其中 feature_name 必须在 FEATURE_NAMES 中。
    """
    data = np.load(path, allow_pickle=True)
    result = {}

    for feat in FEATURE_NAMES:
        key = f"importance_{feat}"
        if key in data:
            # 每个字段是形状 (1,) 的数组
            val = float(np.asarray(data[key]).reshape(-1)[0])
            result[feat] = val
        else:
            # 如果某个特征缺失，可以选择设为 NaN 或者跳过
            # 这里设为 NaN，方便后续在表格中识别
            result[feat] = np.nan

    return result


def main(
    search_dir=".",
    pattern="*_ebm_estimate_*_importance_*.npz",
    output_csv="ebm_feature_importance_all.csv",
):
    """
    在 search_dir 下搜索所有匹配 pattern 的 npz 文件，
    汇总成一个长表 (long table)，每一行是一个 (Dataset, Feature, Kind, Split)。

    也可以很容易 pivot 成宽表。
    """
    glob_pattern = os.path.join(search_dir, pattern)
    files = glob.glob(glob_pattern)
    if not files:
        print(f"在 {search_dir} 中未找到匹配 {pattern} 的 npz 文件")
        return

    print(f"找到 {len(files)} 个 npz 文件，将汇总到 {output_csv}")

    rows = []

    for path in files:
        dataset, kind, split = parse_filename(path)
        print(f"处理: {os.path.basename(path)} -> dataset={dataset}, kind={kind}, split={split}")

        feat2imp = load_importance_npz(path)

        for feat_name, imp in feat2imp.items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Split": split,
                    "Kind": kind,       # 'seal' 或 'label'
                    "Feature": feat_name,
                    "Importance": imp,
                }
            )

    df = pd.DataFrame(rows)
    # 排序
    df = df.sort_values(by=["Dataset", "Split", "Kind", "Feature"], ignore_index=True)

    # 保存
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已写出 {len(df)} 行到 {output_csv}")

    # 如果你更喜欢“每个数据集一行，每个 (Kind, Split, Feature) 一列”，
    # 可以在这里再 pivot 一份。
    wide_output_csv = output_csv.replace(".csv", "_wide.csv")
    df_wide = df.pivot_table(
        index=["Dataset", "Feature"],
        columns=["Kind", "Split"],
        values="Importance",
    )

    # MultiIndex columns 展开为单层列名，比如 'seal_test', 'label_test' 等
    df_wide.columns = [f"{k}_{s}" for k, s in df_wide.columns]
    df_wide = df_wide.reset_index()

    df_wide.to_csv(wide_output_csv, index=False, encoding="utf-8-sig")
    print(f"已额外写出 wide 形式到 {wide_output_csv}")


if __name__ == "__main__":
    # 默认在当前目录搜 *.npz
    main(
        search_dir=".",  # 若 npz 在子目录如 'results' 下，改成 "results"
        pattern="*_ebm_estimate_*_importance_*.npz",
        output_csv="ebm_feature_importance_all.csv",
    )
