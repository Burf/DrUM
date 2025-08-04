import torch

def sim_loss(feature1, feature2, weight = 1, logit_scale = 1, e = 1e-12, reduce = True):
    feature1 = feature1.unsqueeze(1) if feature1.dim() == 2 else feature1
    feature2 = feature2.unsqueeze(1) if feature2.dim() == 2 else feature2
    feature1 = feature1.unsqueeze(1) if feature1.dim() == 3 else feature1
    #feature2 = feature2.unsqueeze(2) if feature2.dim() == 3 else feature2
    feature2 = feature2.unsqueeze(1) if feature2.dim() == 3 else feature2
    sim = (feature1 * feature2).sum(dim = -1) / (feature1.norm(dim = -1) * feature2.norm(dim = -1) + e)
    loss = logit_scale * (1 - sim.mean(dim = -1))
    loss = (loss * weight).mean(dim = -1)
    return torch.clamp_min(loss.mean() if reduce else loss, 0)