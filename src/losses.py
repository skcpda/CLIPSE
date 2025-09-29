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

def sanw_clip_loss(img, txt, logit_scale, mode="debias",
                   alpha=0.5, delta=0.6, lam=4.0, m1=0.3, m2=0.8, gamma=0.05):
    """
    Similarity-Aware Negative Weighting (SANW).
    - mode='debias' or 'bandpass'
    """
    N = img.size(0)
    logits = logit_scale.exp() * img @ txt.t()
    # weights (no grad):
    with torch.no_grad():
        s_mix = _mix_similarity(img, txt, alpha=alpha)
        if mode == "debias":
            w = torch.exp(-lam * torch.relu(s_mix - delta))
        elif mode == "bandpass":
            on  = torch.sigmoid((s_mix - m1) / gamma)
            off = 1.0 - torch.sigmoid((s_mix - m2) / gamma)
            w = on * off
        else:
            raise ValueError("mode must be 'debias' or 'bandpass'")
        w = w + torch.eye(N, device=img.device)

    # row-weighted
    lse_r = torch.logsumexp(logits + torch.log(w + 1e-12), dim=1)
    loss_i2t = (-torch.diag(logits) + lse_r).mean()

    # col-weighted
    lse_c = torch.logsumexp(logits.t() + torch.log(w.t() + 1e-12), dim=1)
    loss_t2i = (-torch.diag(logits.t()) + lse_c).mean()

    return 0.5 * (loss_i2t + loss_t2i)
