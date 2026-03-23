import argparse
import torch
from src.data.dataset import create_dataloaders
from src.models.clip_lora import get_clip_lora
from src.models.loss import ContrastiveLoss
from src.engine.train import train_model
from src.engine.evaluate import evaluate_model
from src.retrieval.faiss_index import FaissRetrievalSystem, get_text_embedding
from transformers import CLIPProcessor

def main():
    parser = argparse.ArgumentParser(description="ScopeSearch: VFX Asset Semantic Retrieval via PEFT CLIP")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "search"], required=True)
    parser.add_argument("--metadata", type=str, help="Path to CSV/JSON metadata for train/eval")
    parser.add_argument("--query", type=str, help="Text query to search for VFX assets")
    parser.add_argument("--epochs", type=int, default=5)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model (with LoRA)
    print("Loading CLIP PEFT Model...")
    model = get_clip_lora()
    
    if args.mode == "train":
        if not args.metadata:
            raise ValueError("--metadata is required for training")
        print("Starting training...")
        train_loader = create_dataloaders(args.metadata, is_csv=args.metadata.endswith(".csv"))
        criterion = ContrastiveLoss()
        model = train_model(model, train_loader, val_dataloader=None, criterion=criterion, num_epochs=args.epochs, device=device)
        print("Training complete.")
        
    elif args.mode == "evaluate":
        if not args.metadata:
            raise ValueError("--metadata is required for evaluation")
        print("Starting evaluation...")
        val_loader = create_dataloaders(args.metadata, is_csv=args.metadata.endswith(".csv"), shuffle=False)
        evaluate_model(model, val_loader, device=device)
        
    elif args.mode == "search":
        if not args.query:
            raise ValueError("--query is required for search mode")
        print(f"Searching for: '{args.query}'...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        embed = get_text_embedding(model, processor, args.query, device=device)
        print("Embeddings generated successfully. System is ready to connect mapped RAG visual database.")

if __name__ == "__main__":
    main()
