#https://github.com/mlfoundations/open_clip
import math
import torch

def AdamW(model, lr = 2.048e-3, weight_decay = 0.2, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, norm = "norm", logit_scale = "logit_scale"):
    model = [model] if not isinstance(model, (tuple, list)) else model
    exclude = lambda n, p: p.ndim < 2 or norm in n or norm in n or "bias" in n or logit_scale in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = []
    for m in model:
        if m is not None:
            named_parameters += list(m.named_parameters() if hasattr(m, "named_parameters") else m)
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias_params, "weight_decay": 0.},
         {"params": rest_params, "weight_decay": weight_decay}],
        lr = lr,
        betas = (beta1, beta2),
        eps = eps)
    return optimizer

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster