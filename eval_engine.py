import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

def evaluate_top1_accuracy(model: nn.Module, dataloader: DataLoader, device: str = 'cpu', num_batches: int = 10) -> float:
    """Evaluates Accuracy on a subset of the validation dataset."""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0

    imagenette_to_imagenet_map = {
        0: 0,    # tench
        1: 217,  # English springer
        2: 482,  # cassette player
        3: 491,  # chain saw
        4: 405,  # church
        5: 566,  # French horn
        6: 569,  # garbage truck
        7: 571,  # gas pump
        8: 574,  # golf ball
        9: 701   # parachute
    }
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # 1. Translate the labels using our dictionary
            true_imagenet_labels = [imagenette_to_imagenet_map[label.item()] for label in labels]
            true_imagenet_labels = torch.tensor(true_imagenet_labels).to(device)

            images, labels = images.to(device), labels.to(device)
            model_dtype = next(model.parameters()).dtype
            images = images.to(model_dtype)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += true_imagenet_labels.size(0)
            correct += (predicted == true_imagenet_labels).sum().item()
            
    return (correct / total) * 100

def build_hardware_simulator(fused_model: nn.Module, hardware_payload: dict) -> nn.Module:
    """
    Takes the FP16 fused model and physically degrades its weights 
    to match the exact INT8/FP8 math that the silicon accelerator will perform.
    """
    print("  [Simulator] Injecting INT8/FP8 Quantization noise into PyTorch model...")
    # Create a copy so we don't destroy the original fused model
    simulated_model = copy.deepcopy(fused_model)
    simulated_model.eval()

    # Dictionary to map PyTorch layer names to module objects
    modules = dict(simulated_model.named_modules())

    with torch.no_grad():
        for layer_name, payload in hardware_payload.items():
            if layer_name in modules and hasattr(modules[layer_name], 'weight'):
                module = modules[layer_name]
                
                # 1. Simulate Weight Quantization (Dequantize the INT8/FP8 back to FP16)
                # Math: W_simulated = W_int8 * W_scale
                w_int8 = payload["quantizedweights_channel"]
                w_scale = payload["weight_scale"]
                
                # Broadcast scale to match weight dimensions
                if w_int8.dim() == 4: # Conv2d
                    scale_reshaped = w_scale.view(-1, 1, 1, 1)
                else:                 # Linear
                    scale_reshaped = w_scale.view(-1, 1)

                # Reconstruct the degraded weights in FP16
                simulated_weight = (w_int8.float() * scale_reshaped.float()).to(module.weight.dtype)
                module.weight.data.copy_(simulated_weight)


    return simulated_model

def verify_full_model_accuracy(raw_model: nn.Module, fused_model: nn.Module, hardware_payload: dict, dataloader: DataLoader):
    """
    Evaluates the Original FP32 Model vs the Simulated INT8/FP8 Hardware Model.
    """
    print("\n" + "=" * 60)
    print(" 🎯 EVALUATING SILICON ACCURACY PARITY (FP32 vs INT8/FP8)")
    print("=" * 60)
    
    print("  Evaluating Original FP32 Model (Golden Baseline)...")
    raw_acc = evaluate_top1_accuracy(raw_model.float(), dataloader, num_batches=20)
    
    # Build the Hardware Simulator using the extracted scales!
    simulated_model = build_hardware_simulator(fused_model, hardware_payload)
    
    print("  Evaluating Simulated Hardware Model (INT8/FP8)...")
    sim_acc = evaluate_top1_accuracy(simulated_model, dataloader, num_batches=20)
    
    accuracy_drop = raw_acc - sim_acc
    
    print(f"\n  Original FP32 Top-1 Accuracy : {raw_acc:.2f}%")
    print(f"  Compiled INT8/FP8 Accuracy   : {sim_acc:.2f}%")
    print(f"  Absolute Accuracy Drift      : {accuracy_drop:.2f}%")
    
    if accuracy_drop < 2.0:
        print("  ✅ RESULT: SILICON COMPILATION SUCCESSFUL (Minimal Degradation)!")
    else:
        print("  ⚠️ RESULT: Significant precision loss detected in INT8/FP8 mapping.")