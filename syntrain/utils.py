import math


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        #if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
        if name.endswith(".bias") or name.endswith("norm.weight"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def adjust_learning_rate(optimizer, it, 
        warmup_iters=1000, lr_decay_iters=80000, lr=1e-3, min_lr=1e-4):
    if it < warmup_iters: # 1) linear warmup for warmup_iters steps
        lr = lr * it / warmup_iters
    elif it > lr_decay_iters: # 2) if it > lr_decay_iters, return min learning rate
        lr = min_lr
    else: # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + (lr - min_lr) * coeff

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
