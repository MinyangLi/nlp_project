from transformers import AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import pickle

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


dataset = 'truthfulQA' #specify the dataset
device = 'cuda:0'
load_path = f"results/{dataset}/prediction.pkl" 
save_path = f"results/{dataset}/correctness.json"

# Sentences we want sentence embeddings for
chunks = load_pickle_file(load_path)
results = [x for chunk in chunks for x in chunk]
model = AutoModelForSequenceClassification.from_pretrained( 
    '/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/hallucination_evaluation_model', trust_remote_code=True) # path to hallucination_evaluation_model
model.eval()
model.to(device)

scores = []
for item in tqdm(results):
    merged_prediction = item[0]['merged_prediction']
    merged_true_answer = item[0]['merged_true_answer']
    score = model.predict([(merged_true_answer, merged_prediction)])[0]
    scores.append(score.item())


with open(save_path, "w") as f:
    json.dump(scores, f) 
