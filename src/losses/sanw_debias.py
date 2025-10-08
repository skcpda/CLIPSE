"""
SANW-Debias loss: Down-weight near-positives / likely false negatives.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _pairwise_cosine(x):
    """Compute pairwise cosine similarities."""
    x = F.normalize(x, dim=-1)
    return x @ x.t()


def sanw_debias_loss(logits_per_image, logits_per_text,
                     img_feats, txt_feats,
                     alpha=0.5, delta=0.6, lam=4.0):
    """
    SANW-Debias loss: Down-weight negatives that are semantically close to anchor.
    
    Args:
        logits_per_image: [B, B] similarity matrix (image -> text)
        logits_per_text: [B, B] similarity matrix (text -> image)
        img_feats: [B, D_img] normalized image features
        txt_feats: [B, D_txt] normalized text features
        alpha: Weight for text similarity vs image similarity
        delta: Threshold for near-positive detection
        lam: Sharpness of sigmoid transformation
    
    Returns:
        loss: Scalar loss tensor
        stats: Dict with weight statistics
    """
    device = logits_per_image.device
    B = logits_per_image.size(0)
    tgt = torch.arange(B, device=device)

    # Auxiliary similarities
    Sii = _pairwise_cosine(img_feats)      # [B, B]
    Stt = _pairwise_cosine(txt_feats)      # [B, B]

    # Affinity & weights in [0,1]
    A = alpha * Stt + (1 - alpha) * Sii
    Wnear = torch.sigmoid(lam * (A - delta))  # near-positive likelihood
    Wneg  = 1.0 - Wnear

    # Build per-row weights for negatives (mask diag)
    neg_mask = (~torch.eye(B, dtype=torch.bool, device=device)).float()
    w_i = Wneg * neg_mask
    
    # Normalize weights per row so total negative mass ≈ B-1
    row_sum = w_i.sum(dim=1, keepdim=True).clamp_min(1e-6)
    w_i = w_i * ((B - 1) / row_sum)

    # Weighted log-softmax for image->text
    logp_i = F.log_softmax(logits_per_image, dim=1)
    # Gather positive term and sum weighted negatives
    pos_i = logp_i[torch.arange(B), tgt]
    # For negatives, we use exp(logp)*w then sum
    neg_i = (torch.exp(logp_i) * w_i).sum(dim=1).clamp_min(1e-12).log()
    loss_i = -(pos_i - neg_i).mean()

    # Symmetric text->image
    w_t = w_i.t()   # symmetry
    logp_t = F.log_softmax(logits_per_text, dim=1)
    pos_t = logp_t[torch.arange(B), tgt]
    neg_t = (torch.exp(logp_t) * w_t).sum(dim=1).clamp_min(1e-12).log()
    loss_t = -(pos_t - neg_t).mean()

    loss = 0.5 * (loss_i + loss_t)

    # Optional stats for logging
    with torch.no_grad():
        w_mean = (Wnear * neg_mask).sum() / (neg_mask.sum() + 1e-12)
        w_low  = ((Wnear * neg_mask) < 1e-3).float().mean()
        w_high = ((Wnear * neg_mask) > 0.8).float().mean()
        stats = dict(
            w_mean=float(w_mean), 
            w_low_pct=float(w_low)*100, 
            w_high_pct=float(w_high)*100
        )

    return loss, stats
