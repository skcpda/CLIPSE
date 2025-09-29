import torch
import torch.nn.functional as F

def clip_ce_loss(img, txt, logit_scale):
    # standard symmetric InfoNCE used in CLIP
    logits = logit_scale.exp() * img @ txt.t()
    labels = torch.arange(img.size(0), device=img.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

@torch.no_grad()
def _mix_similarity(img, txt, alpha=0.5):
    s_tt = (txt @ txt.t()).clamp(-1, 1)
    s_ii = (img @ img.t()).clamp(-1, 1)
    s_mix = alpha * s_tt + (1 - alpha) * s_ii
    s_mix.fill_diagonal_(0.0)
    return s_mix

def sanw_debias_loss(img, txt, logit_scale, alpha=0.5, delta=0.6, lam=4.0, 
                     stopgrad_weights=True, geometry_align_enabled=False):
    """
    SANW Debias mode: Down-weight near-positives to preserve local structure.
    """
    N = img.size(0)
    logits = logit_scale.exp() * img @ txt.t()
    
    # Compute similarity weights
    if stopgrad_weights:
        with torch.no_grad():
            s_mix = _mix_similarity(img, txt, alpha=alpha)
            w = torch.exp(-lam * torch.relu(s_mix - delta))
            w = w + torch.eye(N, device=img.device)
    else:
        s_mix = _mix_similarity(img, txt, alpha=alpha)
        w = torch.exp(-lam * torch.relu(s_mix - delta))
        w = w + torch.eye(N, device=img.device)

    # Weighted log-softmax
    lse_r = torch.logsumexp(logits + torch.log(w + 1e-12), dim=1)
    loss_i2t = (-torch.diag(logits) + lse_r).mean()

    lse_c = torch.logsumexp(logits.t() + torch.log(w.t() + 1e-12), dim=1)
    loss_t2i = (-torch.diag(logits.t()) + lse_c).mean()

    loss = 0.5 * (loss_i2t + loss_t2i)
    
    # Optional geometry alignment
    if geometry_align_enabled:
        img_norm = img / img.norm(dim=-1, keepdim=True)
        txt_norm = txt / txt.norm(dim=-1, keepdim=True)
        geometry_loss = torch.nn.functional.mse_loss(img_norm @ img_norm.t(), txt_norm @ txt_norm.t())
        loss = loss + 0.1 * geometry_loss
    
    return loss

def sanw_bandpass_loss(img, txt, logit_scale, alpha=0.5, m1=0.3, m2=0.8, 
                       gamma=0.05, floor=1e-3, stopgrad_weights=True):
    """
    SANW Bandpass mode: Emphasize hard negatives while avoiding false negatives.
    """
    N = img.size(0)
    logits = logit_scale.exp() * img @ txt.t()
    
    # Compute similarity weights
    if stopgrad_weights:
        with torch.no_grad():
            s_mix = _mix_similarity(img, txt, alpha=alpha)
            on = torch.sigmoid((s_mix - m1) / gamma)
            off = 1.0 - torch.sigmoid((s_mix - m2) / gamma)
            w = on * off
            w = torch.clamp(w, min=float(floor))  # Avoid zero weights
            w = w + torch.eye(N, device=img.device)
    else:
        s_mix = _mix_similarity(img, txt, alpha=alpha)
        on = torch.sigmoid((s_mix - m1) / gamma)
        off = 1.0 - torch.sigmoid((s_mix - m2) / gamma)
        w = on * off
        w = torch.clamp(w, min=float(floor))
        w = w + torch.eye(N, device=img.device)

    # Weighted log-softmax
    lse_r = torch.logsumexp(logits + torch.log(w + 1e-12), dim=1)
    loss_i2t = (-torch.diag(logits) + lse_r).mean()

    lse_c = torch.logsumexp(logits.t() + torch.log(w.t() + 1e-12), dim=1)
    loss_t2i = (-torch.diag(logits.t()) + lse_c).mean()

    return 0.5 * (loss_i2t + loss_t2i)

# Legacy function for backward compatibility
def sanw_clip_loss(img, txt, logit_scale, mode="debias",
                   alpha=0.5, delta=0.6, lam=4.0, m1=0.3, m2=0.8, gamma=0.05):
    """
    Legacy SANW function for backward compatibility.
    """
    if mode == "debias":
        return sanw_debias_loss(img, txt, logit_scale, alpha, delta, lam)
    elif mode == "bandpass":
        return sanw_bandpass_loss(img, txt, logit_scale, alpha, m1, m2, gamma)
    else:
        raise ValueError("mode must be 'debias' or 'bandpass'")
