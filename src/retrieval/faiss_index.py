import faiss
import numpy as np
import torch
from transformers import CLIPProcessor

class FaissRetrievalSystem:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        # Using Inner Product since embeddings will be normalized (equivalent to Cosine Similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.image_paths = []
        
    def add_embeddings(self, embeddings, image_paths):
        """
        Add batch of embeddings to FAISS index.
        embeddings: torch.Tensor or np.ndarray of shape (N, D)
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
            
        # Ensure normalization for IP to act as Cosine Similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.image_paths.extend(image_paths)
        
    def search(self, text_embeddings, top_k=5):
        """
        Search for top_k most similar images for given text embeddings.
        text_embeddings: torch.Tensor or np.ndarray of shape (N, D)
        """
        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.cpu().numpy()
            
        faiss.normalize_L2(text_embeddings)
        distances, indices = self.index.search(text_embeddings, top_k)
        
        results = []
        for i in range(len(indices)):
            query_results = []
            for j in range(top_k):
                idx = indices[i][j]
                if idx != -1 and idx < len(self.image_paths):
                    query_results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(distances[i][j])
                    })
            results.append(query_results)
            
        return results

def get_text_embedding(model, processor, text, device="cuda"):
    model.eval()
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features
