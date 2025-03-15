# TemporalPPR-NodeEmbeddings  
**Enhancing Temporal Graph Embeddings with Personalized PageRank-based Random Walks**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview  
This repository enhances **node property prediction** in temporal graphs by integrating **Personalized PageRank (PPR)** into temporal graph embedding models. Built on the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/), we focus on the **TGBn-trade** dataset to demonstrate how PPR-augmented embeddings improve performance on node classification tasks.

## 🚀 Key Features  
- **PPR-Augmented Embeddings**: Combines temporal node embeddings with global structural signals from PPR.  
- **Snapshot-Specific PPR**: Computes yearly PPR matrices to capture evolving graph structure.  
- **No Learning in P**: PPR matrices are static and interpretable (not trainable).  
- **SDG-Compatible Design**: Ready to integrate incremental PPR updates for large graphs (via [SDG](https://github.com/DongqiFu/SDG)).  
- **TGBn-trade Focus**: Optimized for the smallest TGB dataset, ensuring fast experimentation.      
- **Compatibility** with TGB and modifications of [DyGLib_TGB](https://github.com/yule-BUAA/DyGLib_TGB) codebase.
- **Improved results** on TGBn-trade node property prediction.   
---

## Background  
### Temporal Graph Benchmark (TGB)  
TGB provides datasets and tasks for evaluating models on dynamic graphs. **TGBn-trade** is the smallest dataset, tracking yearly international trade relationships (nodes = countries, edges = trade volumes) over 31 years.  

### Problem  
Existing temporal graph models focus on **local neighborhoods** for embeddings but lack **global structural context**. Our solution enriches embeddings with PPR to capture long-range dependencies.  

---

## Methodology  
### Pipeline  
1. **Yearly PPR Matrices**: For each snapshot (year), compute a PPR matrix where the *i*-th row contains restart probabilities from node *i*.  
2. **Temporal Embeddings**: Generate embeddings using models like DyGFormer.  
3. **PPR Fusion**:  
   - Concatenate embeddings with PPR vectors.  
   - Down-project via a linear layer for classification.  
### Why PPR?  
- Captures **global influence** of nodes beyond immediate neighbors.  
- **Efficient updates** for dynamic graphs (no full recomputation).  

---

## Train  
### Example of training `DyGFormer` on `tgbn-trade` dataset: 
```bash  
python train_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0
