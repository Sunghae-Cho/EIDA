# EIDA: Estimated Intrinsic Dimension Adapter

A parameter-efficient fine-tuning method that leverages PCA-based intrinsic dimension estimation of token representations to construct low-rank adapters tailored to each weight matrix.

## Motivation

LoRA was inspired by prior work on the intrinsic dimensionality of deep learning models ([Li et al., 2018](https://arxiv.org/abs/1804.08838)), which showed that optimization can be effectively performed in a randomly sampled low-dimensional subspace. However, in the fine-tuning setting, we have a fixed starting point (the pretrained model) and access to the target dataset — so rather than using a *random* subspace, we can *estimate* a task-relevant one.

EIDA exploits this by analyzing the distribution of token representations at each weight matrix via uncentered-PCA, then constructing adapters aligned with the estimated low-dimensional structure. This yields a compact, data-informed parameterization of the weight update ΔW.

## Method

For a pretrained weight matrix $W$, let $X$ denote the input token representations and $\Delta W$ the gradient update after one forward-backward pass.

1. **Sample collection**: Pass a subset of the training data through the model and collect token samples $X$ at each weight matrix, as well as the corresponding $\Delta W \cdot X$.

2. **PCA-based projection**: Perform uncentered-PCA on the collected samples to obtain:
   - $A$: a projection from $W$'s input space onto the low-dimensional subspace where $X$ is concentrated
   - $C$: a projection from $W$'s output space onto the low-dimensional subspace where $\Delta W \cdot X$ is concentrated

3. **Adapter construction**: Insert a zero-initialized learnable parameter $B$ between $A$ and $C$, and replace $W$ with:

$$W + C^\top \cdot B \cdot A$$

Since the columns of $C$ are orthonormal ($C \cdot C^\top = I$), $C^\top$ acts as a natural embedding from the subspace back into the full latent space.

<p align="center">
  <img src="./figure/EIDA.png" alt="EIDA architecture" width="500">
</p>

### Adaptive Rank Selection

The subspace dimension (rank) for each weight matrix is chosen based on empirical cosine similarity measurements. Specifically, for each weight position, if the cosine similarity at 64 dimensions exceeds that at 32 dimensions by more than 0.03, the 64-dimensional approximation is used; otherwise, 32 dimensions suffice.

## Experiments

### RoBERTa-base on GLUE SST-2

Token samples were collected from 73 weight matrices (Q, K, V, O, fc1, fc2 across 12 encoder layers + the first classifier weight).

Average cosine similarity between input tokens and their projections onto the estimated subspace
<p align="left">
  <img src="./figure/256_input_0.png" alt="left image" width="35%">
  <img src="./figure/256_input_1.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/256_input_2.png" alt="left image" width="35%">
  <img src="./figure/256_input_3.png" alt="right image" width="35%">
</p>

Average cosine similarity between Δoutput tokens and their projections onto the estimated subspace
<p align="left">
  <img src="./figure/256_delta_output_0.png" alt="left image" width="35%">
  <img src="./figure/256_delta_output_1.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/256_delta_output_2.png" alt="left image" width="35%">
  <img src="./figure/256_delta_output_3.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/256_delta_output_4.png" alt="left image" width="35%">
  <img src="./figure/256_delta_output_5.png" alt="right image" width="35%">
</p>

Fine-tuning was performed for 4 epochs with lr=2e-4, weight decay=0.1, FP16, and a linear schedule (10% warmup).

Fine-tuning result
| Model | Accuracy |
|---|---|
| RoBERTa-base (full fine-tuning) | 94.8 |
| RoBERTa-base + LoRA | 95.1 |
| **RoBERTa-base + EIDA** | **93.4** |

### GPT-2 on E2E NLG Challenge

Token samples were collected from 72 weight matrices (Q, K, V, O, fc1, fc2 across 12 decoder blocks).

Average cosine similarity between input tokens and their projections onto the estimated subspace
<p align="left">
  <img src="./figure/128_input_0.png" alt="left image" width="35%">
  <img src="./figure/128_input_1.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/128_input_2.png" alt="left image" width="35%">
  <img src="./figure/128_input_3.png" alt="right image" width="35%">
</p>

Average cosine similarity between Δoutput tokens and their projections onto the estimated subspace
<p align="left">
  <img src="./figure/128_delta_output_0.png" alt="left image" width="35%">
  <img src="./figure/128_delta_output_1.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/128_delta_output_2.png" alt="left image" width="35%">
  <img src="./figure/128_delta_output_3.png" alt="right image" width="35%">
</p>
<p align="left">
  <img src="./figure/128_delta_output_4.png" alt="left image" width="35%">
  <img src="./figure/128_delta_output_5.png" alt="right image" width="35%">
</p>

Fine-tuning was performed for 16 epochs with lr=5e-5, weight decay=0.01, FP16, and a linear schedule (15% warmup). Generation used beam search with weight 10.

Fine-tuning result
| Model | BLEU | NIST | MET | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| GPT-2 Medium (355M) + LoRA | 70.4 | 8.85 | 46.8 | 71.8 | 2.53 |
| **GPT-2 (124M) + EIDA** | **67.4** | **8.53** | **45.1** | **70.1** | **2.43** |

> Note: EIDA uses the smaller GPT-2 (124M) compared to LoRA's GPT-2 Medium (355M).

## Repository Structure

```
EIDA/
├── EIDA/
│   ├── EIDA.py                  # Adapter classes (Linear_with_adapter, Conv1D_with_adapter)
│   ├── PCA.py                   # uncentered-PCA implementation for token samples
│   ├── reconstruct_roberta.py   # Token sampling during RoBERTa forward pass
│   └── reconstruct_gpt2.py      # Token sampling during GPT-2 forward pass
├── RoBERTa-SST2-graph.ipynb     # Dimension analysis & cosine similarity plots (RoBERTa)
├── RoBERTa-SST2-train.ipynb     # Fine-tuning RoBERTa with EIDA on SST-2
├── GPT2-E2ENLG-graph.ipynb      # Dimension analysis & cosine similarity plots (GPT-2)
├── GPT2-E2ENLG-train.ipynb      # Fine-tuning GPT-2 with EIDA on E2E NLG
└── E2E/                         # E2E NLG Challenge dataset files
```

### Notebooks

| Notebook | Description |
|---|---|
| `RoBERTa-SST2-graph.ipynb` | Collects token samples from RoBERTa-base on SST-2, performs dimensionality reduction, and plots cosine similarity between original tokens and their projections onto estimated subspaces |
| `RoBERTa-SST2-train.ipynb` | Applies EIDA to fine-tune RoBERTa-base on SST-2 and evaluates accuracy |
| `GPT2-E2ENLG-graph.ipynb` | Collects token samples from GPT-2 on E2E NLG, performs dimensionality reduction, and plots cosine similarity |
| `GPT2-E2ENLG-train.ipynb` | Applies EIDA to fine-tune GPT-2 on E2E NLG and generates predictions in the [benchmark format](https://github.com/tuetschek/e2e-dataset) |

## Key Observations

The uncentered-PCA reveals that token representations at each layer are strongly concentrated in low-dimensional subspaces — consistent with findings on anisotropy in contextualized representations ([Ethayarajh, 2019](https://arxiv.org/abs/1909.00512)). Even at 32–64 dimensions, projections preserve high cosine similarity (>0.8 in most layers), confirming that the effective intrinsic dimension is far smaller than the full embedding dimension (768 or 3072).

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838) (Li et al., 2018)
- [How Contextual are Contextualized Word Representations?](https://arxiv.org/abs/1909.00512) (Ethayarajh, 2019)
