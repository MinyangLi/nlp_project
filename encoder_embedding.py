from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
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

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

dataset = 'truthfulQA' # specify the dataset
device = 'cuda:0'
load_path = f"results/{dataset}/prediction.pkl" 

# Sentences we want sentence embeddings for
chunks = load_pickle_file(load_path)
results = [x for chunk in chunks for x in chunk]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/all-mpnet-base-v2') # path to all-mpnet-base-v2
model = AutoModel.from_pretrained('/hpc2hdd/home/mli861/nlp_project/NLI_Project_v2/ckpt/all-mpnet-base-v2', torch_dtype=torch.float16) # path to all-mpnet-base-v2
model.eval()
model.to(device)

pred_embeddings = []
true_embeddings = []
for item in tqdm(results):
    merged_prediction = item[0]['merged_prediction']
    merged_true_answer = item[0]['merged_true_answer']
    sentences = [merged_prediction, merged_true_answer]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    pred_embeddings.append(sentence_embeddings[0])
    true_embeddings.append(sentence_embeddings[1])


save_path = f"results/{dataset}/embeddings.pkl"

save_obj = {
    "pred_embeddings": pred_embeddings,
    "true_embeddings": true_embeddings
}

import pickle
with open(save_path, "wb") as f:
    pickle.dump(save_obj, f)