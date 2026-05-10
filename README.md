# Reproducing Cosine, FS-BGE, and Bidirectional NLI Scores

This repository reproduces three groups of scores used in the report:

1. **Cosine similarity** baseline with all-mpnet-base-v2 sentence embeddings.
2. **FS-BGE** embedding-based factual similarity, including BGE-M3 ColBERT, token reweighting, span-level penalty, and the final FS-BGE score.
3. **Bidirectional NLI** scores, including Forward, Backward/F1-style, and F_beta entailment scores.

## 1. Environment

Python 3.10 is recommended. Install the common dependencies:

```bash
pip install torch transformers datasets sentence-transformers tqdm numpy matplotlib
```

For FS-BGE, also install BGE-M3 support:

```bash
pip install FlagEmbedding
```

GPU is strongly recommended for generation, HHEM/NLI scoring, and BGE-M3 token scoring.

## 2. Download Models

Download these checkpoints to your local machine and replace the hard-coded paths where needed. The original scripts still contain paths such as `/hpc2hdd/...`.

| Model | Used in | Path variable / script |
|---|---|---|
| Llama-3.2-3B-Instruct | answer generation | `llama_Inference.py`: `model_path` |
| Qwen2.5-7B-Instruct | merge Q&A into factual sentence | `llama_Inference.py`: `merge_model_path` |
| all-mpnet-base-v2 | cosine baseline | `encoder_embedding.py` |
| BGE-M3 (`BAAI/bge-m3`) | BGE-M3 ColBERT and FS-BGE | `compute_bge_m3_baseline.py`, `bge_fact_token_score.py` |
| HHEM-2.1-Open | pseudo correctness score | `eval_hem.py` |
| bert-large-NER | optional entity score | `factual_ultils.py`: `NER_CKPT` |
| DeBERTa-v3-large-mnli-fever-anli-ling-wanli | NLI scores | `factual_ultils.py`: `NLI_CKPT` |

For BGE-M3, the FS-BGE scripts try model paths in this order:

1. `ckpt/bge-m3`
2. local Hugging Face cache
3. online model id `BAAI/bge-m3`

Datasets are provided under `processed_data/{truthfulQA, sciq, simple_questions_wiki}/merged_fb.json`.

## 3. Generate Predictions and Correctness Scores

Run the following steps for each dataset: `nq`, `truthfulQA`, `sciq`, `simple_questions_wiki`.

To match the report, use these sample sizes:

| Dataset | `--max_samples` |
|---|---:|
| `nq` | 9183 |
| `truthfulQA` | 817 |
| `sciq` | 5000 |
| `simple_questions_wiki` | 5000 |

### Step 1: Generate predictions

```bash
python llama_Inference.py --dataset nq --max_samples 9183
python llama_Inference.py --dataset truthfulQA --max_samples 817
python llama_Inference.py --dataset sciq --max_samples 5000
python llama_Inference.py --dataset simple_questions_wiki --max_samples 5000
```

Outputs land in `updated_results/<dataset>/prediction.pkl` and `updated_results/<dataset>/prediction.json`.

### Step 2: HHEM correctness scores

`eval_hem.py` currently has a hard-coded `dataset` variable near the top of the file. Set it to each dataset name and run once per dataset:

```bash
python eval_hem.py
```

This produces `updated_results/<dataset>/correctness.json`.

## 4. Cosine Similarity Baseline

`encoder_embedding.py` currently has a hard-coded `dataset` variable. Set it to each dataset name and run once per dataset:

```bash
python encoder_embedding.py
```

This produces `updated_results/<dataset>/embeddings.pkl`.

Then compute cosine similarity for all datasets:

```bash
python compute_sim.py --results-root updated_results
```

This produces `updated_results/<dataset>/embedding_metrics.json`.

Plot cosine results:

```bash
python plot_anything.py --score cos
```

## 5. FS-BGE Embedding Scores

FS-BGE requires `prediction.json` and `correctness.json` from the previous steps.

### Step 1: Compute vanilla BGE-M3 ColBERT scores

```bash
python compute_bge_m3_baseline.py --datasets nq truthfulQA sciq simple_questions_wiki --device cuda:0 --batch-size 16
```

This produces:

```text
updated_results/<dataset>/embedding_metrics_bge_m3.json
```

The key metric is `bge_m3_colbert`.

### Step 2: Compute fact-aware token reweighting

```bash
python bge_fact_token_score.py --datasets nq truthfulQA sciq simple_questions_wiki --device cuda:0 --batch-size 16 --max-length 256 --explain-limit 30
```

This produces:

```text
updated_results/<dataset>/embedding_metrics_bge_m3_fact.json
updated_results/<dataset>/bge_fact_token_explanations.json
```

Important metrics:

- `fact_weighted_colbert`: token reweighting only
- `fact_sensitive_similarity`: token reweighting plus token-level mismatch penalty

### Step 3: Compute final FS-BGE span-aware score

```bash
python fact_span_score.py --datasets nq truthfulQA sciq simple_questions_wiki --explain-limit 30
```

This produces:

```text
updated_results/<dataset>/embedding_metrics_bge_m3_fact_span.json
updated_results/<dataset>/bge_fact_span_explanations.json
```

Important metrics:

- `bge_m3_colbert`: vanilla BGE-M3 ColBERT baseline
- `token_reweighting_similarity`: token reweighting score copied from the token stage
- `span_penalty_similarity`: BGE-M3 ColBERT minus answer-span mismatch penalty
- `fact_span_similarity`: final FS-BGE score

### Step 4: Plot FS-BGE and ablations

```bash
# Vanilla BGE-M3 ColBERT
python plot_anything.py --score bge_m3_colbert

# Only token reweighting
python plot_anything.py --score bge_token_reweighting

# Only answer-span mismatch penalty
python plot_anything.py --score bge_span_penalty

# Final FS-BGE
python plot_anything.py --score fs_bge
```

Figures and printed AUC/Pearson values land in `updated_results/plots_<score>/`.

## 6. Bidirectional NLI Scores

```bash
python compute_hybrid.py --results-root updated_results --entailment-fbeta-beta 0.3
```

This writes the following files into each `updated_results/<dataset>/`:

- `entailment_true_entail_pred_score.json`: Forward score (reference -> prediction)
- `entailment_score.json`: symmetric F1-style score of forward and backward entailment
- `entailment_fbeta_score.json`: F_beta score with `beta = 0.3`

Plot NLI results:

```bash
python plot_anything.py --score entailment_true_only
python plot_anything.py --score entailment_fbeta
python plot_anything.py --score entailment
```

## File Map

| File | Purpose |
|---|---|
| `llama_Inference.py` | Generate predictions and merged factual sentences |
| `eval_hem.py` | HHEM correctness scores |
| `encoder_embedding.py` | all-mpnet sentence embeddings for cosine baseline |
| `compute_sim.py` | Cosine / L1 / L2 from sentence embeddings |
| `compute_bge_m3_baseline.py` | BGE-M3 dense/sparse/ColBERT/hybrid pair scores |
| `bge_fact_token_score.py` | Fact-aware token reweighting and token-level mismatch penalty |
| `fact_span_score.py` | Final FS-BGE score with answer-span mismatch penalty |
| `factual_ultils.py` | NER and bidirectional NLI utilities |
| `compute_hybrid.py` | Forward / Backward / F1 / F_beta entailment scores |
| `plot_anything.py` | AUC, Pearson, distribution / ROC / correlation plots for all supported scores |
