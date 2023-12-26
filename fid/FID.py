# use FID to evaluate my model and ChatGPT

# import os
# import sys
# import numpy as np
# import torch
#
# from datasets import load_dataset
#
# dataset = load_dataset("ZahrizhalAli/mental_health_conversational_dataset")
#
# dataset = dataset['train']['text']
# question = [i.split('<ASSISTANT>: ')[0].replace('<HUMAN>: ', '') for i in dataset]
# answer = [i.split('<ASSISTANT>: ')[1] for i in dataset]
#
# from transformers import AutoTokenizer, AutoModel
# import torch
# from scipy.linalg import sqrtm
# import numpy as np
#
# # Function to calculate FID
# def calculate_fid(mu1, sigma1, mu2, sigma2):
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     covmean = sqrtm(sigma1.dot(sigma2))
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid
#
# # Function to convert text to vectors
# def text_to_vectors(texts, model, tokenizer):
#     model.eval()
#     with torch.no_grad():
#         tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#         outputs = model(**tokens)
#         embeddings = outputs.last_hidden_state.mean(axis=1)
#     return embeddings
#
# def random_change(texts):
#     # change some word in the text to random word
#     random_wordlist = ['apple', 'banana', 'cat', 'dog', 'elephant', 'fish', 'goat', 'horse', 'icecream', 'jellyfish']
#     for i in range(len(texts)):
#         text = texts[i].split()
#         for j in range(len(text)):
#             if np.random.rand() < 0.1:
#                 text[j] = random_wordlist[np.random.randint(0, len(random_wordlist))]
#         texts[i] = ' '.join(text)
#     return texts
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
#
# text_data1 = question
# text_data2 = answer
# # text_data2 = random_change(text_data2)
#
# # Convert text to vectors
# vectors1 = text_to_vectors(text_data1, model, tokenizer).numpy()
# vectors2 = text_to_vectors(text_data2, model, tokenizer).numpy()
#
# # Calculate statistics
# mu1, sigma1 = vectors1.mean(axis=0), np.cov(vectors1, rowvar=False)
# mu2, sigma2 = vectors2.mean(axis=0), np.cov(vectors2, rowvar=False)
#
# # Calculate FID
# fid = calculate_fid(mu1, sigma1, mu2, sigma2)
# print("FID Score:", fid)

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.linalg import sqrtm
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def fid(reference, generated):
    # Function to calculate FID
    def calculate_fid(mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # Function to convert text to vectors
    def text_to_vectors(texts, model, tokenizer):
        model.eval()
        with torch.no_grad():
            tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(axis=1)
        return embeddings

    text_data1 = reference
    text_data2 = generated

    # Convert text to vectors
    vectors1 = text_to_vectors(text_data1, model, tokenizer).numpy()
    vectors2 = text_to_vectors(text_data2, model, tokenizer).numpy()

    # Calculate statistics
    mu1, sigma1 = vectors1.mean(axis=0), np.cov(vectors1, rowvar=False)
    mu2, sigma2 = vectors2.mean(axis=0), np.cov(vectors2, rowvar=False)

    # Calculate FID
    fid = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid