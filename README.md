# TemporalPPR-NodeEmbeddings  
**Enhancing Temporal Graph Embeddings with Personalized PageRank-based Random Walks**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview  
This repository enhances **node property prediction** in temporal graphs by integrating **Personalized PageRank (PPR)** into temporal graph embedding models. Built on the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/), we focus on the **TGBn-trade** dataset to demonstrate how PPR-augmented embeddings improve performance on node classification tasks.

### Key Features  
- 🎯 **PPR-augmented embeddings** combining local and global graph structure.  
- ⏱️ **Efficient PPR updates** inspired by [SDG (Scalable Dynamic Graph Learning)](https://github.com/DongqiFu/SDG).  
- 📈 **Improved results** on TGBn-trade node property prediction.  
- 🧩 **Compatibility** with TGB and modifications of [DyGLib_TGB](https://github.com/yule-BUAA/DyGLib_TGB) codebase.  

---

## Background  
### Temporal Graph Benchmark (TGB)  
TGB provides datasets and tasks for evaluating models on dynamic graphs. **TGBn-trade** is the smallest dataset, tracking yearly international trade relationships (nodes = countries, edges = trade volumes) over 15 years.  

### Problem  
Existing temporal graph models focus on **local neighborhoods** for embeddings but lack **global structural context**. Our solution enriches embeddings with PPR to capture long-range dependencies.  

---

## Methodology  
### Pipeline  
1. **Yearly PPR Matrices**: For each snapshot (year), compute a PPR matrix where the *i*-th row contains restart probabilities from node *i*.  
2. **Temporal Embeddings**: Generate embeddings using models like TGAT or DySAT.  
3. **PPR Fusion**:  
   - Concatenate embeddings with PPR vectors.  
   - Down-project via a linear layer for classification.  

