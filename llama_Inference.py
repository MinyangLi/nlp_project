import argparse
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle

parser = argparse.ArgumentParser(
    description=""
)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument('--dataset', type=str, choices=['sciq', 'simple_questions_wiki',
                                                    'nq', 'truthfulQA'], required=True, help="Dataset to use.")
parser.add_argument('--max_samples', type=int, default=None,
                    help="Optional cap on number of samples to run from the beginning of the dataset.")
args = parser.parse_args()

if args.dataset in ['nq','truthfulQA']:
    word_limit = 100
else:
    word_limit = 10

def load_pickle_file(path): 
    objs = [] 
    with open(path, "rb") as f: 
        while True: 
            try: 
                objs.append(pickle.load(f)) 
            except EOFError: 
                break
    return objs

def append_pickle(obj, path):
    with open(path, "ab") as f:
        pickle.dump(obj, f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def generate_one_token(prompt: str, do_sample: bool = False,
                        temperature: float = 1.0, top_p: float = 1.0):
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        logits = out.logits[:, -1, :]  # [B,V]
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float().argmax(dim=-1, keepdim=True)
                mask = (torch.arange(sorted_probs.size(-1), device=probs.device)[None, :] <= cutoff).to(probs.dtype)
                truncated = sorted_probs * mask
                truncated = truncated / (truncated.sum(dim=-1, keepdim=True) + 1e-12)
                sampled = torch.multinomial(truncated, num_samples=1)
                next_id = sorted_idx.gather(1, sampled)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
    next_id = next_id.squeeze(0).item()
    next_text = tokenizer.decode([next_id], skip_special_tokens=False)

    hidden_states = out.hidden_states  # tuple of (L+1) layers
    final_layer_hidden = hidden_states[-1]  # (B, T, d)
    last_token_embedding = final_layer_hidden[:, -1, :]

    return next_id, next_text, last_token_embedding

def generate_statement(question, answer):
    system_prompt = (
        "You rewrite Question+Answer into exactly one factual sentence. "
        "Output only that sentence. Do not add dialogue, explanations, or extra lines."
    )
    user_prompt = (
        "Convert the following Q&A into one factual sentence.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Sentence:"
    )

    if hasattr(merge_tokenizer, "apply_chat_template"):
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat_text = merge_tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        encoding = merge_tokenizer(chat_text, return_tensors="pt").to(args.device)
    else:
        fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
        encoding = merge_tokenizer(fallback_prompt, return_tensors="pt").to(args.device)

    eos_ids = []
    if merge_tokenizer.eos_token_id is not None:
        eos_ids.append(merge_tokenizer.eos_token_id)
    for token in ["<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>"]:
        token_id = merge_tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            eos_ids.append(token_id)
    # Keep unique ids while preserving order.
    eos_ids = list(dict.fromkeys(eos_ids))
    eos_terminate = eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)

    outputs = merge_model.generate(
        **encoding,
        do_sample=False,
        max_new_tokens=64,
        pad_token_id=merge_tokenizer.pad_token_id,
        eos_token_id=eos_terminate,
    )
    generated = merge_tokenizer.decode(
        outputs[0][encoding["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    # Cut off typical dialogue spillover artifacts.
    for marker in ["\nHuman:", "\nAssistant:", "\nQuestion:", "\nAnswer:", "\nUser:"]:
        if marker in generated:
            generated = generated.split(marker, 1)[0].strip()

    # Enforce single sentence output as a final guardrail.
    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", generated, flags=re.S)
    if sentence_match:
        statement = sentence_match.group(1).strip()
    else:
        statement = generated.split("\n", 1)[0].strip()

    return statement

merge_model_path = '/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/Qwen2.5-7B-Instruct' # path to Qwen2.5-7B
merge_model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
merge_tokenizer = AutoTokenizer.from_pretrained(merge_model_path)
if merge_tokenizer.pad_token is None:
    merge_tokenizer.pad_token = merge_tokenizer.eos_token
    merge_model.resize_token_embeddings(len(merge_tokenizer))
merge_model.eval()
merge_model.to(args.device)


model_path = '/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/Llama-3.2-3B-Instruct' # path to Llama-3.2-3B-Instruct

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
   tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)
model.eval()

data_path = f"processed_data/{args.dataset}/merged_fb.json"
dataset = load_dataset("json", data_files={"test": data_path})
print(len(dataset['test']))

eval_data = dataset["test"] # by default the whole dataset will be processed.

if args.max_samples is not None:
    capped = min(args.max_samples, len(eval_data))
    eval_data = eval_data.select(range(capped))
    print(f"Capped to first {capped} samples.")

save_path = f"updated_results/{args.dataset}/prediction.pkl" 
json_save_path = os.path.splitext(save_path)[0] + ".json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

results = []
try:
    chunks = load_pickle_file(save_path)
    results = [x for chunk in chunks for x in chunk]
    begin = len(results)
    total_samples = len(eval_data)
    if begin < total_samples:
        eval_data = eval_data.select(range(begin, total_samples))
        print(f"Loaded {begin} previous results from {save_path}")
    else:
        # Nothing left to generate for current capped/full dataset size.
        eval_data = eval_data.select([])
        print(
            f"Loaded {begin} previous results from {save_path}; "
            f"no remaining samples in current run (size={total_samples})."
        )
except FileNotFoundError:
    print("No previous results found. Starting fresh.")

chunk = []
with torch.no_grad():
    for idx, example in enumerate(tqdm(eval_data)):
        output = []
        question = example["question"]

        if args.dataset in ['nq','truthfulQA']:
            prompt = "Answer the following question using exactly one sentence.\nQuestion: " + question + "\nAnswer:"
        else:
            prompt = "Answer the following question using exactly one word. Do not explain.\nQuestion: " + question + "\nAnswer:"
        
        init_enc = tokenizer(prompt, return_tensors="pt")
        init_enc = {k: v.to(args.device) for k, v in init_enc.items()}
        init_input_ids = init_enc["input_ids"]  # [1, T]
        init_tokens = [tokenizer.decode([tid], skip_special_tokens=False)
                    for tid in init_input_ids[0].tolist()]

        generated_tokens = []
        for w in range(word_limit):
            next_id, next_txt, embedding = generate_one_token(
                prompt=prompt, do_sample=False, temperature=1.0, top_p=1.0
            )

            # 结束符
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            if next_txt.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>", "."}:
                break

            print(next_txt)
            generated_tokens.append(next_txt)
            prompt+=next_txt
        
        prediction = ''.join(generated_tokens)
        prediction_length = len(generated_tokens)
        merged_prediction = generate_statement(question, prediction)
        merged_true_answer = generate_statement(question, example["correct_answer"])
        output.append({'question': question,
                       'prediction': prediction,
                       'prediction_length': prediction_length,
                        'true_answer': example["correct_answer"],
                        'merged_prediction': merged_prediction,
                        'merged_true_answer': merged_true_answer})

        chunk.append(output)
        results.append(output)

        if (idx + 1) % 50 == 0:
            append_pickle(chunk, save_path)
            save_json(results, json_save_path)
            chunk.clear()

if len(chunk) > 0:
    append_pickle(chunk, save_path)
    save_json(results, json_save_path)
    chunk.clear()
else:
    # Keep JSON in sync even when no new sample was generated.
    save_json(results, json_save_path)