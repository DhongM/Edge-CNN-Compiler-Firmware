"""
inference_transfer.py
=====================
Handles per-inference PCIe transfer of input images.
Dynamically reads the manifest.json to configure the On-Chip Conversion Unit.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from typing import Union, List

# =============================================================
# PRODUCTION PREPROCESSING PIPELINE
# =============================================================
# Matches the exact transforms used during calibration!
PRODUCTION_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),              # uint8 -> float32 [0,1]
    transforms.Normalize(               # -> float32 [-2.1,+2.6]
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess single image for inference. Returns float32 tensor."""
    image = Image.open(image_path).convert('RGB')
    img_fp32 = PRODUCTION_TRANSFORMS(image)
    return img_fp32.unsqueeze(0)        # [1,3,224,224]

# =============================================================
# READ HARDWARE FIRMWARE MANIFEST
# =============================================================
def load_hardware_config(manifest_path: str) -> dict:
    """Extracts the first layer's Calibration data from the JSON manifest."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # We only need the very first layer's calibration data for the input image!
    first_layer_calib = manifest["calibration_layers"][0]
    
    return {
        "scale": first_layer_calib["input_scale"],
        "zp": first_layer_calib["input_zero_point"]
    }

# =============================================================
# CONVERT TO FP16 FOR PCIe TRANSFER
# =============================================================
def prepare_for_pcie(image_fp32: torch.Tensor, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Casts float32 to FP16 to physically halve the PCIe bandwidth."""
    return image_fp32.to(target_dtype)

# =============================================================
# VERIFY CONVERSION ACCURACY
# =============================================================
def verify_fp16_conversion(image_fp32: torch.Tensor, image_fp16: torch.Tensor, QS_Input_ref: float) -> dict:
    """
    Verify FP16 conversion rounding error does not shift the INT8 bin.
    max_error MUST be << QS_Input to guarantee hardware accuracy.
    """
    diff    = (image_fp32 - image_fp16.float()).abs()
    max_err = diff.max().item()
    
    error_vs_scale = max_err / QS_Input_ref

    print(f"\n [Validation] FP16 PCIe Conversion Check:")
    print(f"    float32 range : [{image_fp32.min():.4f}, {image_fp32.max():.4f}]")
    print(f"    fp16    range : [{image_fp16.float().min():.4f}, {image_fp16.float().max():.4f}]")
    print(f"    max error     : {max_err:.6f}")
    print(f"    vs QS_Input   : {error_vs_scale:.4f}x ({'✅ Safe' if error_vs_scale < 0.1 else '❌ DANGER: Precision Loss'})")

# =============================================================
# SIMULATE ON-CHIP HARDWARE QUANTIZATION
# =============================================================
def simulate_accelerator_input_quantization(image_fp16: torch.Tensor, QS_Input: float, ZP_Input: int) -> torch.Tensor:
    """
    Simulates the physical logic gates of the On-Chip Conversion Unit.
    Equation: clamp(round(FP16 / Scale) + ZP, -128, 127)
    """
    X_fp32    = image_fp16.float() # Upcast to guarantee Python rounding matches silicon
    X_scaled  = X_fp32 / QS_Input
    X_rounded = X_scaled.round()
    X_shifted = X_rounded + ZP_Input
    X_int8    = X_shifted.clamp(-128, 127).to(torch.int8)
    return X_int8

# =============================================================
# COMPLETE INFERENCE PIPELINE
# =============================================================
def run_edge_inference(image_path: str, manifest_path: str):
    print("\n" + "=" * 60)
    print(" 📡 Booting Edge Inference Engine")
    print("=" * 60)

    # 1. Load the Hardware Config dynamically
    print("[1/4] Loading Hardware configuration from manifest...")
    config = load_hardware_config(manifest_path)
    print(f"      -> Hardware expects Scale: {config['scale']:.6f}, ZP: {config['zp']}")

    # 2. Host CPU Preprocessing (FP32)
    print(f"\n[2/4] CPU Preprocessing Image (FP32)...")
    image_fp32 = preprocess_image(image_path)
    print(f"      -> CPU Buffer Size : {image_fp32.numel() * 4:,} bytes")

    # 3. Cast to FP16 and Verify
    print(f"\n[3/4] Casting to FP16 for PCIe Transfer...")
    image_fp16 = prepare_for_pcie(image_fp32, torch.float16)
    verify_fp16_conversion(image_fp32, image_fp16, config['scale'])
    print(f"      -> PCIe Payload    : {image_fp16.numel() * 2:,} bytes (50% Bandwidth Saved!)")

    # 4. Fire over PCIe and simulate Hardware Conversion
    print(f"\n[4/4] Firing data over PCIe to On-Chip Conversion Unit...")
    X_int8 = simulate_accelerator_input_quantization(image_fp16, config['scale'], config['zp'])
    
    print(f"\n 🎉 SUCCESS: Data loaded into Hardware I-SRAM!")
    print(f"      -> Final Shape : {tuple(X_int8.shape)}")
    print(f"      -> Final Dtype : {X_int8.dtype}")
    print(f"      -> Final Range : [{X_int8.min().item()}, {X_int8.max().item()}]")
    print(f"      -> Ready for Systolic Array Matrix Math.")

# =============================================================
# EXECUTION
# =============================================================
if __name__ == "__main__":
    
    # Use the manifest we just generated!
    manifest_file = "./hardware_bins/ResNet_manifest.json"
    
    # Grab a real image from your dataset (Update this filename to one that exists in your folder!)
    # Or, if you want to test right away without an image, you can use the dummy generator below.
    test_image = "./datasets/imagenette2-160/val/n01440764/ILSVRC2012_val_00010267.JPEG" 
    
    if os.path.exists(test_image):
        run_edge_inference(image_path=test_image, manifest_path=manifest_file)
    else:
        print("⚠️ Real image not found. Running with simulated camera frame...")
        # Create a dummy image file for testing
        dummy_img = Image.fromarray(torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy())
        dummy_img.save("simulated_camera.jpg")
        run_edge_inference(image_path="simulated_camera.jpg", manifest_path=manifest_file)