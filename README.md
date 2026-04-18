## Introduction

The project is designed to investigate whether the **embedding similarity between predictions and correct answers** can properly indicate **prediction correctness**.

## Preparation

### Download the Following Language Models

Please download the following models to your local device:

1. **Llama-3.2-3B-Instruct**  
https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct/files  

This model is used as the **base model for inference**.

2. **Qwen2.5-7B-Instruct**  
https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/files  

This model merges the **response from the base model and its corresponding query** into a **statement for evaluation**.

3. **HHEM-2.1-Open**  
https://huggingface.co/vectara/hallucination_evaluation_model  

This is a **hallucination evaluation model**. It produces scores between **0 and 1** to indicate whether statements are hallucinated given the ground truth.

4. **all-mpnet-base-v2**  
https://huggingface.co/sentence-transformers/all-mpnet-base-v2  

This is an **embedding model** that encodes text into vector representations.

After downloading the models, please **replace the model locations in the scripts** with your local paths.


### Create an Environment

The project is mainly based on **LlamaFactory**:  
https://github.com/hiyouga/LlamaFactory

Please create a **virtual environment** containing the required packages. Also check that your environment includes other necessary components, such as:

- seaborn (for visualization)
- torch
- sentence-transformers
- other dependencies required by LlamaFactory



## Run the Scripts

### Run Inference

Run `llama_inference.py` to conduct inference on the following four datasets:

- nq
- sciq
- simple_questions_wiki
- truthfulQA

The data is stored in the `processed_data` folder.

You may decide how many samples to process for each dataset, as long as the number is **sufficient to support your conclusions**.


### Evaluate Predictions

Run `eval_hem.py` to determine whether the predictions are **correct or hallucinated**.

### Compute Embeddings

Run `encoder_embedding.py` to compute embeddings for:

- predicted statements
- ground-truth statements