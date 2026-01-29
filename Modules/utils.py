

import torch
import torch.nn as nn

def print_model_param_stats(model: nn.Module):
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    print(f"{'Module':40} {'Total':>12} {'Trainable':>12} {'Frozen':>12}")
    print("-" * 80)

    # Iterate over top-level modules
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = module_total - module_trainable

        total_params += module_total
        trainable_params += module_trainable
        frozen_params += module_frozen

        print(f"{name:40} {module_total:12,} {module_trainable:12,} {module_frozen:12,}")

    print("-" * 80)
    print(f"{'TOTAL':40} {total_params:12,} {trainable_params:12,} {frozen_params:12,}")
