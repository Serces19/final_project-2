import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

class ScopeSearchDataset(Dataset):
    """
    Dataset for VFX assets semantic retrieval.
    Expects a metadata file (CSV/JSON/Parquet) with an 'image_path' and 'description' column.
    """
    def __init__(self, metadata_path, processor_name="openai/clip-vit-base-patch32", max_length=77, is_csv=True):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(processor_name)
        self.max_length = max_length
        
        if is_csv:
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.read_json(metadata_path)
            
        assert "image_path" in self.metadata.columns, "Metadata must contain 'image_path' column"
        assert "description" in self.metadata.columns, "Metadata must contain 'description' column"
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = row["image_path"]
        text = str(row["description"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Handle corrupted images or missing files gracefully
            print(f"Error loading image {image_path}: {e}")
            # Fallback to pure black image for robustness during training
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        # Process both image and text
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length,
            truncation=True
        )
        
        # HuggingFace processor adds a batch dimension, we squeeze it
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "image_path": image_path  # Useful for tracking/RAG later
        }

def create_dataloaders(metadata_path, batch_size=32, num_workers=4, is_csv=True, processor_name="openai/clip-vit-base-patch32", shuffle=True):
    dataset = ScopeSearchDataset(
        metadata_path=metadata_path, 
        processor_name=processor_name,
        is_csv=is_csv
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
