"""
SANW-Bandpass loss: Emphasize mid-hard negatives; avoid trivially easy & near-positive.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _pairwise_cosine(x):
    """Compute pairwise cosine similarities."""
    x = F.normalize(x, dim=-1)
    return x @ x.t()


def sanw_bandpass_loss(logits_per_image, logits_per_text,
                       img_feats, txt_feats,
                       alpha=0.5, m1=0.3, m2=0.8, gamma=0.05):
    """
    SANW-Bandpass loss: Emphasize mid-hard negatives.
    
    Args:
        logits_per_image: [B, B] similarity matrix (image -> text)
        logits_per_text: [B, B] similarity matrix (text -> image)
        img_feats: [B, D_img] normalized image features
        txt_feats: [B, D_txt] normalized text features
        alpha: Weight for text similarity vs image similarity
        m1: Lower threshold for bandpass filter
        m2: Upper threshold for bandpass filter
        gamma: Sharpness of sigmoid transitions
    
    Returns:
        loss: Scalar loss tensor
        stats: Dict with weight statistics
    """
    device = logits_per_image.device
    B = logits_per_image.size(0)
    tgt = torch.arange(B, device=device)

    # Auxiliary similarities
    Sii = _pairwise_cosine(img_feats)
    Stt = _pairwise_cosine(txt_feats)

    # Affinity matrix
    A = alpha * Stt + (1 - alpha) * Sii
    
    # Bandpass filter: emphasize mid-hard negatives
    g_low  = torch.sigmoid((A - m1) / gamma)      # suppress too easy
    g_high = 1.0 - torch.sigmoid((A - m2) / gamma)  # suppress near-positive
    Wbp = (g_low * g_high).clamp(0, 1)

    # Apply to negative mask
    neg_mask = (~torch.eye(B, dtype=torch.bool, device=device)).float()
    w_i = Wbp * neg_mask

    # Row-normalize weights to keep scale stable
    row_sum = w_i.sum(dim=1, keepdim=True).clamp_min(1e-6)
    w_i = w_i * ((B - 1) / row_sum)

    # Weighted NCE
    logp_i = F.log_softmax(logits_per_image, dim=1)
    pos_i = logp_i[torch.arange(B), tgt]
    neg_i = (torch.exp(logp_i) * w_i).sum(dim=1).clamp_min(1e-12).log()
    loss_i = -(pos_i - neg_i).mean()

    # Symmetric text->image
    w_t = w_i.t()
    logp_t = F.log_softmax(logits_per_text, dim=1)
    pos_t = logp_t[torch.arange(B), tgt]
    neg_t = (torch.exp(logp_t) * w_t).sum(dim=1).clamp_min(1e-12).log()
    loss_t = -(pos_t - neg_t).mean()

    loss = 0.5 * (loss_i + loss_t)

    # Stats for logging
    with torch.no_grad():
        w_mean = (Wbp * neg_mask).sum() / (neg_mask.sum() + 1e-12)
        band_cov = (Wbp * neg_mask > 0.1).float().mean()
        stats = dict(
            w_mean=float(w_mean), 
            band_coverage_pct=float(band_cov)*100
        )

    return loss, stats
