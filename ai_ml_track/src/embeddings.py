from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained model for generating sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to get normalized embeddings for a list of texts using the pre-trained model
def get_embeddings(texts):
    return model.encode(texts, normalize_embeddings=True)

# Function to compute class centroids by averaging the embeddings of example texts for each category
def class_embeddings(grouped_examples):
    class_embeddings = {}

    for label, texts in grouped_examples.items():
        embeddings = get_embeddings(texts)
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        class_embeddings[label] = centroid

    return class_embeddings