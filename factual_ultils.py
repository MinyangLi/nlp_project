import re
import math
from functools import lru_cache

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NER_CKPT = "/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/bert-large-NER"
NLI_CKPT = "/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(NER_CKPT)
    model = AutoModelForTokenClassification.from_pretrained(NER_CKPT)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if DEVICE.type == "cuda" else -1,
    )


@lru_cache(maxsize=1)
def _get_nli_components():
    tokenizer = AutoTokenizer.from_pretrained(NLI_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_CKPT).to(DEVICE)
    model.eval()

    entailment_idx = None
    for idx, label in model.config.id2label.items():
        if "entail" in label.lower():
            entailment_idx = idx
            break
    if entailment_idx is None:
        raise ValueError(f"Cannot find entailment label in id2label: {model.config.id2label}")

    return tokenizer, model, entailment_idx


def extract_entities(text: str):
    ner = _get_ner_pipeline()
    entities = ner(text)
    return {_normalize_text(ent["word"]) for ent in entities if ent.get("word")}


def _chunk_indices(total_size: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return range(0, total_size, batch_size)


def extract_entities_batch(
    texts,
    batch_size: int = 32,
    show_progress: bool = False,
    progress_desc: str = "NER",
):
    ner = _get_ner_pipeline()
    all_entities = []
    indices = _chunk_indices(len(texts), batch_size)
    if show_progress:
        indices = tqdm(indices, total=(len(texts) + batch_size - 1) // batch_size, desc=progress_desc)

    for start in indices:
        batch_texts = texts[start : start + batch_size]
        batch_outputs = ner(batch_texts, batch_size=batch_size)

        for sample_output in batch_outputs:
            entities = {_normalize_text(ent["word"]) for ent in sample_output if ent.get("word")}
            all_entities.append(entities)

    return all_entities


def _entailment_score(premise: str, hypothesis: str) -> float:
    tokenizer, model, entailment_idx = _get_nli_components()
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1)
    return float(probs[entailment_idx].item())


def _entailment_scores_batch(
    premises,
    hypotheses,
    batch_size: int = 16,
    show_progress: bool = False,
    progress_desc: str = "NLI",
):
    if len(premises) != len(hypotheses):
        raise ValueError(
            f"premises and hypotheses lengths differ: {len(premises)} vs {len(hypotheses)}"
        )

    tokenizer, model, entailment_idx = _get_nli_components()
    scores = []
    indices = _chunk_indices(len(premises), batch_size)
    if show_progress:
        indices = tqdm(indices, total=(len(premises) + batch_size - 1) // batch_size, desc=progress_desc)

    with torch.no_grad():
        for start in indices:
            batch_premises = premises[start : start + batch_size]
            batch_hypotheses = hypotheses[start : start + batch_size]
            inputs = tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, entailment_idx]
            scores.extend(float(x) for x in probs.detach().cpu().tolist())

    return scores


def _f_beta_score(a: float, b: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * a + b
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * a * b / denominator


def compute_factual_scores_test(predicted_answer: str, true_answer: str):
    entities_pred = extract_entities(predicted_answer)
    entities_true = extract_entities(true_answer)

    entity_match = entities_pred & entities_true
    entity_matching_score = len(entity_match) / len(entities_true) if entities_true else 0.0

    true_entail_pred = _entailment_score(true_answer, predicted_answer)
    pred_entail_true = _entailment_score(predicted_answer, true_answer)

    if true_entail_pred + pred_entail_true == 0:
        entailment_f1 = 0.0
    else:
        entailment_f1 = 2 * true_entail_pred * pred_entail_true / (true_entail_pred + pred_entail_true)

    return entity_matching_score, entailment_f1, entities_pred, entities_true, true_entail_pred, pred_entail_true

def compute_factual_scores(predicted_answer: str, true_answer: str):
    entities_pred = extract_entities(predicted_answer)
    entities_true = extract_entities(true_answer)

    entity_match = entities_pred & entities_true
    entity_matching_score = len(entity_match) / len(entities_true) if entities_true else 0.0

    true_entail_pred = _entailment_score(true_answer, predicted_answer)
    pred_entail_true = _entailment_score(predicted_answer, true_answer)

    if true_entail_pred + pred_entail_true == 0:
        entailment_f1 = 0.0
    else:
        entailment_f1 = 2 * true_entail_pred * pred_entail_true / (true_entail_pred + pred_entail_true)

    return entity_matching_score, entailment_f1


def compute_factual_scores_batch(
    predicted_answers,
    true_answers,
    ner_batch_size: int = 32,
    nli_batch_size: int = 16,
    show_progress: bool = False,
    progress_prefix: str = "",
    fbeta_beta: float = 1.0,
    compute_entity: bool = True,
):
    if len(predicted_answers) != len(true_answers):
        raise ValueError(
            f"predicted_answers and true_answers lengths differ: "
            f"{len(predicted_answers)} vs {len(true_answers)}"
        )

    prefix = f"{progress_prefix} " if progress_prefix else ""
    entities_pred = None
    entities_true = None
    if compute_entity:
        entities_pred = extract_entities_batch(
            predicted_answers,
            batch_size=ner_batch_size,
            show_progress=show_progress,
            progress_desc=f"{prefix}NER(pred)",
        )
        entities_true = extract_entities_batch(
            true_answers,
            batch_size=ner_batch_size,
            show_progress=show_progress,
            progress_desc=f"{prefix}NER(true)",
        )
    true_entail_pred = _entailment_scores_batch(
        true_answers,
        predicted_answers,
        batch_size=nli_batch_size,
        show_progress=show_progress,
        progress_desc=f"{prefix}NLI(true->pred)",
    )
    pred_entail_true = _entailment_scores_batch(
        predicted_answers,
        true_answers,
        batch_size=nli_batch_size,
        show_progress=show_progress,
        progress_desc=f"{prefix}NLI(pred->true)",
    )

    scores = []
    for idx, (t2p, p2t) in enumerate(zip(true_entail_pred, pred_entail_true)):
        if compute_entity:
            e_pred = entities_pred[idx]
            e_true = entities_true[idx]
            entity_match = e_pred & e_true
            entity_matching_score = len(entity_match) / len(e_true) if e_true else 0.0
        else:
            entity_matching_score = 0.0
        if t2p + p2t == 0:
            entailment_f1 = 0.0
        else:
            entailment_f1 = 2 * t2p * p2t / (t2p + p2t)

        entailment_true_only = p2t
        entailment_square_sqrt = t2p * t2p + math.sqrt(max(p2t, 0.0))
        entailment_fbeta = _f_beta_score(
            entailment_true_only,
            entailment_square_sqrt,
            fbeta_beta,
        )

        scores.append(
            {
                "entity_matching_score": float(entity_matching_score),
                "entailment_f1": float(entailment_f1),
                "true_entail_pred": float(entailment_true_only),
                "pred_entail_true": float(p2t),
                "entailment_square_sqrt": float(entailment_square_sqrt),
                "entailment_fbeta": float(entailment_fbeta),
            }
        )
    return scores


def compute_factual_scores_ablation(predicted_answer: str, true_answer: str):
    entities_pred = extract_entities(predicted_answer)
    entities_true = extract_entities(true_answer)

    entity_match = entities_pred & entities_true
    entity_matching_score = len(entity_match) / len(entities_true) if entities_true else 0.0

    true_entail_pred = _entailment_score(true_answer, predicted_answer)
    return entity_matching_score, true_entail_pred


def compute_factual_scores_ablation_batch(
    predicted_answers,
    true_answers,
    ner_batch_size: int = 32,
    nli_batch_size: int = 16,
    show_progress: bool = False,
    progress_prefix: str = "",
):
    if len(predicted_answers) != len(true_answers):
        raise ValueError(
            f"predicted_answers and true_answers lengths differ: "
            f"{len(predicted_answers)} vs {len(true_answers)}"
        )

    prefix = f"{progress_prefix} " if progress_prefix else ""
    entities_pred = extract_entities_batch(
        predicted_answers,
        batch_size=ner_batch_size,
        show_progress=show_progress,
        progress_desc=f"{prefix}NER(pred)",
    )
    entities_true = extract_entities_batch(
        true_answers,
        batch_size=ner_batch_size,
        show_progress=show_progress,
        progress_desc=f"{prefix}NER(true)",
    )
    true_entail_pred = _entailment_scores_batch(
        true_answers,
        predicted_answers,
        batch_size=nli_batch_size,
        show_progress=show_progress,
        progress_desc=f"{prefix}NLI(true->pred)",
    )

    scores = []
    for e_pred, e_true, t2p in zip(entities_pred, entities_true, true_entail_pred):
        entity_match = e_pred & e_true
        entity_matching_score = len(entity_match) / len(e_true) if e_true else 0.0
        scores.append((entity_matching_score, t2p))
    return scores


if __name__ == "__main__":
    predicted =  "This project was finished in Illinois"
    truth = "This project was finished in Chicago"
    entity_score, entailment_f1, entities_pred, entities_true, true_entail_pred, pred_entail_true = compute_factual_scores_test(predicted, truth)
    print("entity_matching_score:", entity_score)
    print("entailment_f1:", entailment_f1)
    print("entities_pred:", entities_pred)
    print("entities_true:", entities_true)
    print("true_entail_pred:", true_entail_pred)
    print("pred_entail_true:", pred_entail_true)