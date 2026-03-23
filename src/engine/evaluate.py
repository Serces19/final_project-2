import torch
import torch.nn.functional as F
from tqdm import tqdm

def calculate_recall_at_k(similarity_matrix, k=1):
    """
    similarity_matrix: [num_images, num_texts]
    For each image-text pair, checks if the correct text matches within top k
    """
    # Text-to-Image retrieval
    # similarity_matrix[i, j] matches Image i with Text j.
    # We transpose to get Texts as rows, Images as columns for Text-to-Image search
    text_to_image_sim = similarity_matrix.t()
    
    num_queries = text_to_image_sim.size(0)
    top_k_indices = torch.topk(text_to_image_sim, k=k, dim=1).indices
    
    # Ground truth: query i matches image i
    targets = torch.arange(num_queries, device=similarity_matrix.device).view(-1, 1)
    
    # Check if target is in top_k
    hits = (top_k_indices == targets).sum().item()
    return hits / num_queries

def calculate_mrr(similarity_matrix):
    text_to_image_sim = similarity_matrix.t()
    num_queries = text_to_image_sim.size(0)
    
    # Sort indices by similarity descending
    sorted_indices = torch.argsort(text_to_image_sim, dim=1, descending=True)
    targets = torch.arange(num_queries, device=similarity_matrix.device).view(-1, 1)
    
    # Get ranks (1-based index where the target is found)
    ranks = (sorted_indices == targets).nonzero(as_tuple=True)[1] + 1
    
    mrr = (1.0 / ranks.float()).mean().item()
    return mrr

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    
    print("Extracting features for evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
            
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)
            
    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Compute similarity matrix (num_samples x num_samples)
    print("Computing metrics...")
    similarity_matrix = all_image_embeds @ all_text_embeds.t()
    
    r1 = calculate_recall_at_k(similarity_matrix, k=1)
    r5 = calculate_recall_at_k(similarity_matrix, k=5)
    r10 = calculate_recall_at_k(similarity_matrix, k=10)
    mrr = calculate_mrr(similarity_matrix)
    
    results = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "MRR": mrr
    }
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    return results
