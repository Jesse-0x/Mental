import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.linalg import sqrtm
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def kl(reference, generated):
    # Function to calculate FID
    def kl_divergence(p_matrix, q_matrix):
        """
        Calculate the Kullback-Leibler divergence between corresponding rows
        of two matrices. Each row is treated as a separate distribution.

        Parameters:
        - p_matrix: A 2D numpy array representing the first set of distributions.
        - q_matrix: A 2D numpy array representing the second set of distributions.

        Returns:
        - A numpy array of KL divergence values for each corresponding row.
        """
        # Apply softmax to each row to get probability distributions
        p_matrix = softmax(p_matrix, axis=1)
        q_matrix = softmax(q_matrix, axis=1)

        # Compute the KL divergence for each row
        kl_divergences = np.array([entropy(p_row, q_row) for p_row, q_row in zip(p_matrix, q_matrix)])
        return kl_divergences

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

    # Calculate KL divergence
    kl = kl_divergence(vectors1, vectors2)
    return kl



