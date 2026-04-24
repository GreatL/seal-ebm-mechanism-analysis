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
