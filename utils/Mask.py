import torch

def maskGenerator(src, target):
    target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
    target_seq_len = target.size(1)
    future_restrict_mask = (1 - torch.triu(torch.ones(1, target_seq_len, target_seq_len), diagonal=1))
    target_mask = target_mask & future_restrict_mask.bool()

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    return src_mask, target_mask       