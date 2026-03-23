import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

def get_clip_lora(model_name="openai/clip-vit-base-patch32", r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Loads a base CLIP model and applies LoRA (Low-Rank Adaptation) to both the Vision 
    and Text encoders to efficiently adapt it to the VFX post-production domain.
    """
    # Load base model
    model = CLIPModel.from_pretrained(model_name)
    
    # We apply LoRA to the attention layers (q_proj, v_proj) of both vision and text models
    # Different HuggingFace CLIP versions might name these modules differently. 
    # Usually they are named 'q_proj', 'k_proj', 'v_proj', 'out_proj'.
    target_modules = ["q_proj", "v_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # Apply PEFT
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    
    return peft_model
