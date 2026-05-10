# Reproducing Cosine Similarity & Bidirectional NLI Scores

This repository reproduces the **Cosine similarity** baseline (Table 1) and the **Forward / Backward / F_beta** entailment scores (Table 2) reported in our report. (FS-BGE is not included in this codebase.)

## 1. Environment

Install the following packages (Python 3.10 recommended):

```bash
pip install torch transformers datasets sentence-transformers tqdm numpy matplotlib
```

GPU is required for inference and HHEM/NLI scoring.

## 2. Download Models

Download these checkpoints to your local machine and replace the hard-coded paths in the scripts (search for `/hpc2hdd/...`):

| Model | Used in | Path variable to update |
|---|---|---|
| Llama-3.2-3B-Instruct | `llama_Inference.py` | `model_path` |
| Qwen2.5-7B-Instruct | `llama_Inference.py` | `merge_model_path` |
| all-mpnet-base-v2 | `encoder_embedding.py` | `AutoTokenizer/AutoModel.from_pretrained(...)` |
| HHEM-2.1-Open (`vectara/hallucination_evaluation_model`) | `eval_hem.py` | model load path |
| bert-large-NER (`dslim/bert-large-NER`) | `factual_ultils.py` | `NER_CKPT` |
| DeBERTa-v3-large-mnli-fever-anli-ling-wanli (`MoritzLaurer/...`) | `factual_ultils.py` | `NLI_CKPT` |

Datasets are provided under `processed_data/{truthfulQA, sciq, simple_questions_wiki}/merged_fb.json`.

> **Note on `processed_data/nq/merged_fb.json`**: this file (~67 MB) is excluded from the repository via `.gitignore`. Regenerate it from the official Natural Questions source and place it under `processed_data/nq/merged_fb.json` before running the NQ pipeline.

## 3. Reproduce the Pipeline

Run the following four steps **for each of the four datasets**: `nq`, `truthfulQA`, `sciq`, `simple_questions_wiki`.

To match the report, use these sample sizes:

| Dataset | `--max_samples` |
|---|---|
| `nq` | 9183 |
| `truthfulQA` | 817 |
| `sciq` | 5000 |
| `simple_questions_wiki` | 5000 |

### Step 1 — Generate predictions

```bash
python llama_Inference.py --dataset nq --max_samples 9183
python llama_Inference.py --dataset truthfulQA --max_samples 817
python llama_Inference.py --dataset sciq --max_samples 5000
python llama_Inference.py --dataset simple_questions_wiki --max_samples 5000
```

Outputs land in `updated_results/<dataset>/prediction.{pkl,json}`.

### Step 2 — HHEM correctness scores (pseudo ground-truth)

`eval_hem.py` has a hard-coded `dataset` variable at the top of the file. Set it to each dataset name and run once per dataset:

```bash
python eval_hem.py   # after editing the `dataset` variable
```

Produces `updated_results/<dataset>/correctness.json`.

### Step 3 — Sentence embeddings (for cosine similarity)

`encoder_embedding.py` also has a hard-coded `dataset` variable. Set it to each dataset name and run once per dataset:

```bash
python encoder_embedding.py   # after editing the `dataset` variable
```

Produces `updated_results/<dataset>/embeddings.pkl`.

Then compute cosine similarity (handles all datasets in one call):

```bash
python compute_sim.py --results-root updated_results
```

Produces `updated_results/<dataset>/embedding_metrics.json`.

### Step 4 — Bidirectional entailment scores (Forward / Backward / F1 / F_beta)

```bash
python compute_hybrid.py --results-root updated_results --entailment-fbeta-beta 0.3
```

This writes the following files into each `updated_results/<dataset>/`:

- `entailment_true_entail_pred_score.json` — **Forward score** (reference → prediction)
- `entailment_score.json` — Symmetric F1 of forward + backward
- `entailment_fbeta_score.json` — **F_beta score** with `beta = 0.3` 

The **Backward score** (prediction → reference) is computed internally; if you need it as a standalone JSON, save `pred_entail_true` from `factual_ultils.compute_factual_scores_batch`.

## 4. Generate Tables and Figures

`plot_anything.py` computes ROC-AUC and Pearson correlation, and saves distribution / ROC / correlation figures (2×2 layout over the 4 datasets). Use `--score` to pick which score to evaluate:

```bash
# Cosine similarity 
python plot_anything.py --score cos

# Forward score 
python plot_anything.py --score entailment_true_only

# F_beta score with beta=0.3 
python plot_anything.py --score entailment_fbeta

# Symmetric F1 entailment
python plot_anything.py --score entailment
```

Figures and printed AUC values land in `updated_results/plots_<score>/`.

The AUC / Pearson values printed by `plot_anything.py` for `cos`, `entailment_true_only`, and `entailment_fbeta` correspond to the **Cosine similarity** row of Table 1 and the **Forward score** / **F_beta score** rows of Table 2 in the report.

## File Map

| File | Purpose |
|---|---|
| `llama_Inference.py` | Generate predictions + merged factual sentences |
| `eval_hem.py` | HHEM correctness scores (pseudo-labels) |
| `encoder_embedding.py` | mpnet sentence embeddings |
| `compute_sim.py` | Cosine / L1 / L2 from embeddings |
| `factual_ultils.py` | NER + bidirectional NLI utilities |
| `compute_hybrid.py` | Forward / Backward / F1 / F_beta entailment scores |
| `plot_anything.py` | AUC, Pearson, distribution / ROC / correlation plots |
