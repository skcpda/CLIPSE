"""
Baseline InfoNCE loss for CLIP training.
"""
import torch
import torch.nn.functional as F


def clip_ce_loss(logits_per_image: torch.Tensor,
                 logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Standard InfoNCE loss for CLIP training.
    
    Args:
        logits_per_image: [B, B] similarity matrix (image -> text)
        logits_per_text: [B, B] similarity matrix (text -> image)
    
    Returns:
        Scalar loss tensor
    """
    B = logits_per_image.size(0)
    targets = torch.arange(B, device=logits_per_image.device)
    
    # Cross-entropy loss for both directions
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    
    return 0.5 * (loss_i + loss_t)
