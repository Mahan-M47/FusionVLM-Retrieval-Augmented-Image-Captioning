
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

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
    

def add_dict(main, new):
    for k, v in new.items():
        main.setdefault(k, []).append(v)
        
        
def same_architecture(model_a, model_b):
    keys_a = list(model_a.state_dict().keys())
    keys_b = list(model_b.state_dict().keys())
    return keys_a == keys_b

def same_shapes(model_a, model_b):
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    for k in sd_a:
        if sd_a[k].shape != sd_b[k].shape:
            print(f"Shape mismatch at {k}: {sd_a[k].shape} vs {sd_b[k].shape}")
            return False
    return True

def same_weights(model_a, model_b):
    for (ka, va), (kb, vb) in zip(
        model_a.state_dict().items(),
        model_b.state_dict().items()
    ):
        if not torch.equal(va, vb):
            print(f"Mismatch at {ka}")
            return False
    return True


def display_image(image, title=''):    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off') 
    plt.title(title)
    plt.show()