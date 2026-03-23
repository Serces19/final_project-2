import torch
import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass: extract features
        # CLIP model output contains 'image_embeds' and 'text_embeds'
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Calculate Contrastive Loss
        loss = criterion(image_embeds, text_embeds)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

def train_model(model, train_dataloader, val_dataloader, criterion, num_epochs=5, learning_rate=1e-4, device="cuda"):
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Only optimize trainable parameters (LoRA layers + anything else un-frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch)
        # Note: In practice, we evaluate Recall@K / MRR here using val_dataloader
            
    print("Training complete.")
    return model
