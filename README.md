# TemporalPPR-NodeEmbeddings  
**Enhancing Temporal Graph Embeddings with Personalized PageRank-based Random Walks**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📄 Overview  
This repository enhances **node property prediction** in temporal graphs by integrating **Personalized PageRank (PPR)** into temporal graph embedding models. Built on the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/), we focus on the **TGBn-trade** dataset to demonstrate how PPR-augmented embeddings improve performance on node classification tasks.

## 💡 Key Features  
- **PPR-Augmented Embeddings**: Combines temporal node embeddings with global structural signals from PPR.  
- **Snapshot-Specific PPR**: Computes yearly PPR matrices to capture evolving graph structure.  
- **No Learning in P**: PPR matrices are static and interpretable (not trainable).  
- **SDG-Compatible Design**: Ready to integrate incremental PPR updates for large graphs (via [SDG](https://github.com/DongqiFu/SDG)).  
- **TGBn-trade Focus**: Optimized for the smallest TGB dataset, ensuring fast experimentation.      
- **Compatibility** with TGB and modifications of [DyGLib_TGB](https://github.com/yule-BUAA/DyGLib_TGB) codebase.
- **Improved results** on TGBn-trade node property prediction.   
---

## 🔄 Background  
### Temporal Graph Benchmark (TGB)  
TGB provides datasets and tasks for evaluating models on dynamic graphs. **TGBn-trade** is the smallest Dynamic Node Property Prediction dataset, tracking yearly international trade relationships (`nodes` = **countries**, `edges` = **trade volumes**) over 31 years.  
### PPR (P) Matrix
A row-stochastic matrix $`P \in \mathbb{R}^{n \times n}`$. Each entry $`P_{i,j}`$ represents the probability of reaching node $`j`$ from node $`i`$ via teleporting random walks (with a restart probability $`\alpha = 0.15 `$).
### Problem  
Existing temporal graph models focus on **local neighborhoods** for embeddings but lack **global structural context**. Our solution enriches embeddings with PPR to capture long-range dependencies.  

---

## 📖 Methodology  
### Pipeline  
1. **Yearly PPR Matrices**: For each snapshot (year), compute a PPR matrix where the *i*-th row contains restart probabilities from node *i*.  
2. **Temporal Embeddings**: Generate embeddings using models like DyGFormer.  
3. **PPR Fusion**:  
   - Concatenate embeddings with PPR vectors.  
   - Down-project via a linear layer for classification.  
### Why PPR?  
- **Global Perspective**: PPR captures long-range dependencies beyond immediate neighbors.
- **No Learning in P**: PPR matrices are computed statically for each snapshot (year), ensuring structural signals remain fixed and interpretable.
- **Scalability**: Inspired by SDG, PPR computation leverages sparse operations for large graphs. 

---

## 📊 Results

<p align="center"><strong>Evaluation of Different Models Before and After Applying the Proposed Method:</strong></p>
| Model Name   | Model Accuracy Before Applying Proposed Method | Model Accuracy After Applying Proposed Method | Improvement Percentage |
|:--------------:|:-----------------------------------------------:|:---------------------------------------------:|:----------------------:|
| DygFormer    | 0.388 ± 0.006                                  | 0.4002 ± 0.0037                              | 1.22%                  |
| TGN          | 0.374 ± 0.001                                  | 0.3877 ± 0.0032                             | 1.3%                   |
| DyRep        | 0.374 ± 0.001                                  | 0.3910 ± 0.0011                             | 1.7%                   |
| TCL          | 0.3743 ± 0.005                                | 0.3913 ± 0.0007                             | 1.7%                   |
| Graphmixer   | 0.3747 ± 0.0013                                | 0.3924 ± 0.0008                             | 1.77%                  |
| TGAT         | 0.3741 ± 0.0003                                | 0.3895 ± 0.0012                             | 1.54%                  |

---

## 📦 Requirement 
- **PyTorch**
- **py-tgb**
- **numpy**
- **scipy**
- **tqdm**
---

## 🛠️ Train  
### Example of training `DyGFormer` on `tgbn-trade` dataset: 
```bash  
python train_node_classification.py --dataset_name tgbn-trade --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0
```


### 3. Interpretability:
Rows directly encode node-specific influence distributions.

## Role in Prediction:
- Augments embeddings with structural importance (e.g., a country’s global trade reach).
- Combined with temporal embeddings via concatenation and linear projection.
