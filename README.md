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

## Data Format

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
