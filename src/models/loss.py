import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Standard Contrastive Loss for CLIP as defined in the project manifesto.
    Minimizes the distance between paired image and text embeddings, and maximizes 
    the distance with incorrect descriptions.
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        # temperature (tau) is often learned, but can also be fixed
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Calculate cosine similarity scaled by temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        # Ground truth labels are the diagonal elements (i-th image matches i-th text)
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Loss for image -> text and text -> image
        loss_i = self.loss_fn(logits_per_image, labels)
        loss_t = self.loss_fn(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
