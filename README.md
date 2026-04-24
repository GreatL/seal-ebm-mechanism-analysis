# SEAL-EBM Mechanism Analysis

Code for the paper:

> **Data-driven link formation in complex networks: a statistical–mechanical analysis of subgraph-based graph neural networks**  
> (submitted to EPJ Data Science)

This repository contains the code used to train the SEAL link prediction model, extract structural features on enclosing subgraphs, fit global surrogate models (EBMs and decision trees), and perform ensemble-level analyses of link formation mechanisms across 27 real-world networks.

The code is organised as a set of scripts that reproduce the main steps and figures in the paper.

---

## 1. Environment and Dependencies

Tested with:

- Python 3.10+
- PyTorch (>= 1.12)  
- PyTorch Geometric (matching your PyTorch / CUDA version)  
- NetworkX (>= 2.8)  
- SciPy  
- NumPy  
- pandas  
- scikit-learn  
- matplotlib  
- [interpret](https://github.com/interpretml/interpret) (for Explainable Boosting Machines)

You can start from something like:

```bash
pip install torch torch-geometric networkx scipy numpy pandas scikit-learn matplotlib interpret
```

Please ensure that `torch-geometric` and `torch` versions are compatible with your CUDA / CPU setup; refer to the PyTorch Geometric installation guide for details.

## 2. Data Format

The scripts assume that each network is provided as an edge list in a plain text file:

- One undirected edge per line:
`u v`
- Nodes are represented as integers (node IDs will be re-indexed internally if necessary).
Default assumption:
```text
data/
  ADV.txt
  BUP.txt
  CDM.txt
  ...
  ZWL.txt
```
You can adjust the edge-path templates (e.g. `EDGE_PATH_TEMPLAT`E in `run_randomization_batch.py` or the `--edge_path argument`) if your files are stored elsewhere.

The real-world datasets used in the paper are available from:

- NOESIS link prediction benchmark: https://noesis.ikor.org/datasets/link-prediction
- The sources referenced in the original SEAL paper [Zhang & Chen, NeurIPS 2018].

## 3. Training SEAL from an Edge List
The main SEAL training script is:
- `src/seal_from_edgelist.py`
This script:
1. Loads an edge list file;
2. Constructs 2-hop enclosing subgraphs for candidate edges;
3. Splits edges into train/validation/test sets;
4. Trains a DGCNN-based SEAL model;
5. Saves:
   - a cache file with subgraph data (*_cache.pt);
   - a trained model checkpoint in models/.
Example:
```bash
cd src

python seal_from_edgelist.py \
  --edge_path ../data/Celegans.txt \
  --num_hops 2 \
  --val_ratio 0.10 \
  --test_ratio 0.20 \
  --seed 12345 \
  --model_dir ../models \
  --save_model

```
This will train SEAL on the Celegans network and save the cache and model for later use.

## 4. Extracting Structural Features and Training EBMs
Once SEAL is trained and the cache file is available, you can extract structural features from enclosing subgraphs and train EBMs as global surrogates using:
- `src/explain_seal_gam_new.py`
This script:
- Loads the SEAL cache and model;
- For each subgraph in a given split (e.g. `test`), computes a 18-dimensional structural feature vector:
  -  `n_sub`, `m_sub`, `avg_deg`, `avg_clust`
  - `deg_u`, `deg_v`, `deg_min`, `deg_max`, `deg_sum`, `deg_prod`
  - `CN`, `AA`, `RA`, `Jaccard`, `sp_len`, `tri_u`, `tri_v`, `tri_uv`
- Collects:
  - SEAL outputs (scores/probabilities);
  - ground-truth labels;
- Trains:
  - `EBM_estimate_SEAL` (regressor) to approximate SEAL outputs;
  - `EBM_estimate_label` (classifier) to approximate ground-truth labels;
- Saves global explanations (shape functions and importance) and sample-level data in .npz files.

Example:
```bash
cd src

python explain_seal_gam_new.py \
  --edge_path ../data/Celegans.txt \
  --num_hops 2 \
  --val_ratio 0.10 \
  --test_ratio 0.20 \
  --seed 12345 \
  --model_dir ../models \
  --model_name FBK_model.pt \
  --split test \
  --batch_size 64 \
  --max_samples 5000 \
  --interactions 0

```
This will generate:
- `Celegans_ebm_estimate_seal_global_test.npz`
- `Celegans_ebm_estimate_label_global_test.npz`
- `Celegans_ebm_samples_test.npz`
which are used by subsequent scripts.

## 5. Aggregating Feature Importances Across Networks
After running `explain_seal_gam_new.py` for all 27 networks (or any subset), you can aggregate the global feature importances:
```bash
cd src

python export_ebm_importance_to_csv.py \
  --search_dir . \
  --pattern "*_ebm_estimate_*_importance_test.npz" \
  --output_csv ebm_feature_importance_all.csv

```
Then you can:

Generate the LaTeX table of mean/std importances and average ranks, and other analyses using:
- `src/analyze_ebm_results.py`
Example:
```bash
python analyze_ebm_results.py \
  --result_dir . \
  --split test \
  --do_table

```
This prints a LaTeX table similar to Table 3 in the paper and saves it to `ebm_global_importance_stats_test.tex`.

You can also:
- Plot shape functions for selected features:
```bash
python analyze_ebm_results.py \
  --result_dir . \
  --split test \
  --do_shapes \
  --dataset_for_shapes Celegans \
  --features_for_shapes sp_len Jaccard tri_uv deg_u
```
- Extract case-study examples:
```bash
python analyze_ebm_results.py \
  --result_dir . \
  --split test \
  --do_cases \
  --dataset_for_cases Celegans
```
## 6. Ensemble-Level PCA / t-SNE Visualisation

To replicate the PCA and t-SNE embeddings of the mechanism signatures (Figure 4), use:
- `src/ensemble_visualization.py`
This script expects `ebm_feature_importance_all_wide.csv` (from `export_ebm_importance_to_csv.py`) and generates PCA and t-SNE plots coloured by network domain.
```bash
python ensemble_visualization.py
```
Outputs:
- `ensemble_pca_label_long.pdf/png`
- `ensemble_tsne_label_long.pdf/png`

## 7. Scaling Relations (triangles vs clustering, degree vs heterogeneity)

To generate the scaling plots (triangle importance vs global clustering, degree importance vs degree heterogeneity, cf. Figure 5), use:
- `src/scaling_relations_long.py`
- Alternatively, `src/compute_correlations.py` to compute and print Spearman/Pearson statistics.
Both scripts require:
- `ebm_feature_importance_all_wide.csv`
- `network_stats.csv` containing per-network `Dataset`, `global_clustering`, `degree_std`, etc.
Example:
```bash
python scaling_relations_long.py
python compute_correlations.py
```
## 8. Degree-Preserving Randomisation Experiments
To reproduce the randomisation experiments (Figure 6):
1. Set your datasets and edge-path template in:
  - `src/run_randomization_batch.py`
2. Run batch randomisation:
```bash
python run_randomization_batch.py
```
This script:
- Reads each network from `EDGE_PATH_TEMPLATE`;
- Trains a Label–EBM on the original graph and on a degree-preserving randomised counterpart;
- Saves feature importances and AUC/AP to `randomization_importances.csv`.
3. Plot summarised results using:
  - `src/plot_randomization_summary.py`
```bash
python plot_randomization_summary.py

```
This generates:
- Per-dataset bar plots and line plots comparing original vs randomised feature importances.
`randomization_experiment_simple.py` provides a minimal standalone example of the same idea for a single dataset.

## 9. Decision Tree Surrogates for Rule Extraction
To fit shallow decision trees that approximate SEAL’s outputs and extract human-readable if–then rules, use:
- `src/explain_seal_tree.py`
Example:
```bash
python explain_seal_tree.py \
  --edge_path ../data/Celegans.txt \
  --num_hops 2 \
  --val_ratio 0.10 \
  --test_ratio 0.20 \
  --seed 12345 \
  --model_dir ../models \
  --model_name Celegans_model.pt \
  --split test \
  --batch_size 64 \
  --max_samples 5000 \
  --tree_max_depth 3 \
  --tree_min_samples_leaf 50

```
This script outputs:
- A text file FBK_seal_tree_rules.txt containing the decision tree rules;
- Feature importances for the tree.
These rules correspond to the qualitative case studies discussed in the paper.

## 10. Reproducibility and Extensions
- The repository contains all scripts used to generate the tables and figures in the paper.
- By following the steps above (SEAL training → feature extraction → EBM/decision-tree training → aggregation and plotting), you should be able to reproduce all reported results.

The surrogate-based analysis framework is model-agnostic and could in principle be applied to other subgraph-aware link prediction architectures (e.g., ELPH, BUDDY) by replacing the SEAL training stage and reusing the same structural feature extraction and surrogate fitting pipeline.

If you have any questions or encounter issues running the code, please open an issue on this repository.
