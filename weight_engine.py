import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Tuple
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization import prepare, convert
import torchvision.models as models

def quant_max_min(Intbitwidth:int):
    quant_max=(1<<(Intbitwidth-1))-1
    quant_min=-(1<<(Intbitwidth-1))
    return quant_max, quant_min

def tensors_scale_symmetric(tensor_fp:torch.Tensor, Intbitwidth:int, dim=None): 
    """Symmetric per-tensor or per-channel scale computation (zero_point=0)."""
    # for zeropoint of tensor =0, scale=tensorfp_max/float(quan_max)
    # max value 

    if dim is None:
      # Per-Tensor: Finds the single largest number in the whole block
      # Note: tensorfp_max is the same precsion with tensor_fp!
        tensorfp_max= tensor_fp.abs().max()
    else:
      # Per-Channel: Finds the largest number in each specified dimension
        tensorfp_max= tensor_fp.abs().max(dim=dim,keepdim=True)[0]
        
    tensorfp_max=torch.clamp(tensorfp_max, min=1e-5)
      
    quan_max, _=quant_max_min(Intbitwidth)

    # Because tensorfp_max is FP16/BF16, dividing by float keeps it FP16/BF16!  
    scale=tensorfp_max/float(quan_max)
    
    zero_point=0
      #scale per output chanel
    return scale, zero_point

def tensor_quantized(tensor_fp:torch.Tensor, scale, zero_point, Intbitwidth:int, dtype: torch.dtype=torch.int8):  
    
    """
    Dequantize: single fp_tensor from tensor_fp = (tensor_quantized - zero_point) * scale ,  
    Quantize:  tensor_quantized = int(round(tensor_fp / scale)) + zero_point  and return tensor_quantized 
    """
   
    #1. Allow FP32, FP16, and BF16
    assert tensor_fp.is_floating_point(), "Input tensor must be a floating point type"
     
    if isinstance(scale, torch.Tensor): 
        try:
            torch.broadcast_shapes(scale.shape, tensor_fp.shape)
        except RuntimeError:
            raise AssertionError(
            f"scale shape{scale.shape}not broadcastiable "
            f"with tensor shape {tensor_fp.shape}"
            )
            
    if isinstance(zero_point, torch.Tensor):
        try:
            torch.broadcast_shapes(zero_point.shape, tensor_fp.shape)
        except RuntimeError:
            raise AssertionError(
                f"zero_point shape {zero_point.shape} not broadcastable "
                f"with tensor shape {tensor_fp.shape}"
            )   
        
        
    quan_max, quan_min=quant_max_min(Intbitwidth)
    scaledtensor=tensor_fp/scale
    tensor_round =scaledtensor.round()
    
    #tensor_before=tensor_round.to(int)
    
    quantizedtensor=tensor_round+zero_point
      
    quantizedtensor=quantizedtensor.clamp(min=quan_min, max=quan_max)
    
    return quantizedtensor.to(dtype)
    

def quantize_weights_per_channel(weight:torch.Tensor, Intbitwidth: int=8): 
    #using each scale for each output channel to increase accuray 
    #MLP layer weight[cout, cin], CNN layer weight[cout,cin, wh, ww]
   
    orginal_dtype=weight.dtype
    original_shape = weight.shape
    out_channels=weight.shape[0]
     
    # 1. Flatten to 2D [out_channels, everything_else]
    weight_flat=weight.view(out_channels,-1)
     
    # 2. Call the general function, but explicitly pass dim=1!
    # Per-channel scale from flattened weight → shape [out_ch, 1]
    # Get scale (shape: [out_channels, 1])
    weight_scale_flat, zero_point = tensors_scale_symmetric(weight_flat, Intbitwidth, dim=1)
    #print(f" weight scale flat shape : {weight_scale_flat.shape}")
     
    # 3. convert weight_scale to the same dtype of weight tenor 
    # Flatten scales to exactly 1D so the registers can read them!  shape [out_ch, 1]
    weight_scale_flat_fl=weight_scale_flat.to(orginal_dtype)
     
     # 4.Quantize using the 2D scale (broadcasts correctly over flattened weight)
     #  Quantize the weights
    quantized_flat=tensor_quantized(weight_flat, weight_scale_flat_fl,zero_point=zero_point, 
                                    Intbitwidth=Intbitwidth, dtype=torch.int8) 

    #   Linear result:  weight_scaled.shape = [out_ch, 1]
    #   Conv2d result:  weight_scaled.shape = [out_ch, 1, 1, 1]
    
    # 5. Reshape quantized weights back to original hardware shape (e.g., [64, 64, 3, 3])
    quantizedweights_channel=quantized_flat.view(original_shape)
    
    
    zero_point = torch.zeros(out_channels, dtype=torch.int8)
    

    return  quantizedweights_channel, weight_scale_flat_fl.flatten(),  zero_point 


def convert_fc_weight_to_fp8(module: nn.Linear, layer_name: str, fp8_format: str='FP8_E4M3') -> dict:
    """
    Convert fc Linear weight FP16/BF16 → FP8 per output channel.
    fc is NOT quantized to INT8 (accuracy sensitive).
    FP8 E4M3 max = 448.0   ← better precision, use for weights
    FP8 E5M2 max = 57344.0 ← wider range, use for gradients
    """  
    assert isinstance(module, nn.Linear), f"Expected nn.Linear, got {type(module)}"
  
    if fp8_format == 'FP8_E4M3':
        fp8_max = 448.0
        fp8_dtype = torch.float8_e4m3fn
        safe_guard = 1e-4  # Perfect for E4M3
    elif fp8_format == 'FP8_E5M2':
        fp8_max = 57344.0
        fp8_dtype = torch.float8_e5m2  # Fixed: No 'fn' at the end for e5m2 in PyTorch
        safe_guard = 1e-2  # Perfect for E5M2
    else:
        raise ValueError(f"Unknown fp8 format: {fp8_format}")        
 
    # ── Use float32 for stable computation ───────────
    original_dtype = module.weight.dtype
    Weight = module.weight.data.float() # [out_ch, in_ch]
    
    out_channels = Weight.shape[0]    
    
    # ── Per-channel conversion scale ──────────────────────────
    Weight_flat = Weight.view(out_channels, -1)           # [out_ch, in_ch]
    Weight_max = Weight_flat.abs().max(dim=1).values      # [out_ch]
    Weight_max = Weight_max.clamp(min=safe_guard)               # guard zero
    Weight_fp_scale = Weight_max / fp8_max                # [out_ch] float32

    # ── Convert weight to FP8 ──────────────────────────────────
    Weight_fp_scale_match = Weight_fp_scale.view(-1, 1)   # [out_ch, 1]
    Weight_Converted = Weight_flat / Weight_fp_scale_match # scale to FP8 range
    Weight_fp8 = Weight_Converted.to(fp8_dtype)           # cast to FP8
    
    # ── Convert weight fp8 scale to fp16/bf16 for PCIe ─────────
    Weight_scale_fp16 = Weight_fp_scale.to(original_dtype) # [out_ch] fp16 or bf16
    
    # ── Bias stays fp16/bf16 ───────────────────────────────────
    bias_fp16 = module.bias.data.to(original_dtype) if module.bias is not None else None
                
    fc_weight_result = {
        'layer_name':     layer_name,
        'Weight_fp8':     Weight_fp8,         # FP8, [out_ch, in_ch]
        'weight_scale':   Weight_scale_fp16,  # fp16/bf16, [out_ch]
        'bias_fp16':      bias_fp16,          # fp16/bf16, [out_ch] or None
        'fp8_format':     fp8_format,
        'zero_point':     0,
        'original_shape': module.weight.shape,
        'original_dtype': original_dtype,
    }

    print(f"\n  fc layer FP8 conversion:")
    print(f"    Format      : FP8 {fp8_format.upper()}  (max={fp8_max})")
    print(f"    Weight_fp8  : {tuple(Weight_fp8.shape)}  {Weight_fp8.dtype}") 
    print(f"    weight_scale: {tuple(Weight_scale_fp16.shape)}  {Weight_scale_fp16.dtype}")
    print(f"    scale range : {Weight_scale_fp16.float().min():.6f}, {Weight_scale_fp16.float().max():.6f}")
    
    # FIXED: Safely print the integer without calling Tensor methods
    print(f"    Weight ZP   : type{type(fc_weight_result['zero_point'])} (Value: {fc_weight_result['zero_point']})")

    if bias_fp16 is not None:
        print(f"    bias_fp16   : {tuple(bias_fp16.shape)}  {bias_fp16.dtype}")
    print(f"    PCIe size   : {Weight_fp8.numel()}B (FP8) + {Weight_scale_fp16.numel()*2}B (scale)")

    return fc_weight_result


def full_weight_quantization_model(
       model:nn.Module,
       dtype: torch.dtype=torch.float16,
       fp8_format: str='FP8_E4M3')->dict:
    # ── Step 1: Model Evaluation ──────────────────────────────────
    print("\nStep 1: model set to eval mode")
    model.eval()
    
    # ── Step 2: Convert dtype ──────────────────────────────────
    orginal_model=model.to(dtype)
    print(f"  Weight model type {dtype} ")  
    
    # ── Step 3: Mixed-Precision Weight Quantization ──────────────────────────────────
   
    print("\n" + "=" * 60)
    print("Mixed-Precision Weight Extraction Summary")
    print("=" * 60)
    
    Weight_Quantized_Param = {}
    QUANT_BLACKLIST=['fc','classifier','head'] # Layers to keep in FP8

    for name, module in orginal_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # --- BRANCH 1: The Head Layer (FP8 Path) ---
            if any(top_name in name for top_name in QUANT_BLACKLIST ):
               print(f"  [FP8 PATH] Layer: {name:<10}")
               
               fp8_results = convert_fc_weight_to_fp8(module, layer_name=name, fp8_format=fp8_format)   

               Weight_Quantized_Param[name] = {
                    "type" :  fp8_format, 
                     "quantizedweights_channel": fp8_results['Weight_fp8'],
                     "weight_scale": fp8_results['weight_scale'],
                     "zero_point": 0,
                     "bias":  fp8_results['bias_fp16']
                     }
        # --- BRANCH 2: Hidden Layers (INT8 Path) ---
            else:  
                w_int8, w_scales, w_zps = quantize_weights_per_channel(
                weight=module.weight.data, 
                Intbitwidth=8) 
                bias=module.bias.data.to(dtype) if module.bias is not None else None
                Weight_Quantized_Param[name] = {
                 "type": w_int8.dtype,
                 "quantizedweights_channel": w_int8,
                 "weight_scale": w_scales,
                 "zero_point": w_zps,
                 "bias":  bias
                }
      
                print(f"  [INT8 PATH] Layer: {name:<10}")
                print(f"     └─ Weights: {w_int8.dtype} {tuple(w_int8.shape)}")
                print(f"     └─ Weight Scales : {w_scales.dtype} {tuple(w_scales.shape)}, min: {w_scales.min().item():.6f}, max: {w_scales.max().item():.6f}")
                print(f"     └─ Weight ZPs    : {w_zps.dtype} {tuple(w_zps.shape)}, min: {w_zps.min().item()}, max: {w_zps.max().item()}")

                #print(f"     └─ bias  : {bias.dtype}")
    # ---------------------------------------------------------
    # LAYER 1 INSPECTION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" RAW HARDWARE BYTES INSPECTION: Layer 'conv1'")
    print("=" * 60)
    
    layer_name = 'conv1'
    if layer_name in Weight_Quantized_Param:
        layer_data = Weight_Quantized_Param[layer_name]
        
        w_int8 = layer_data["quantizedweights_channel"]
        scales = layer_data["weight_scale"]
        zps = layer_data["zero_point"]
        
        print(f" [Data Types Match Expected Hardware Target]")
        print(f"    ├─ Weights : {w_int8.dtype} (Shape: {tuple(w_int8.shape)})")
        print(f"    ├─ Scales  : {scales.dtype} (Shape: {tuple(scales.shape)})")
        print(f"    └─ ZPs     : {zps.dtype} (Shape: {tuple(zps.shape)})")
        
        print(f"\n [Memory Payload - First Output Channel]")
        print(f"    ├─ First 5 INT8 Weights : {w_int8[0, 0, 0, :5].tolist()}")
        print(f"    ├─ First 5 Scales       : {scales[:5].tolist()}")
        print(f"    └─ First 5 Zero Points  : {zps[:5].tolist()}")
 
    else:
        print(f" Layer '{layer_name}' not found in the quantized dictionary.")    
    return Weight_Quantized_Param
    
# =============================================================
# SECTION 7: DEMO
# =============================================================


if __name__ == "__main__":
    # 1. Load a pre-trained model (e.g., ResNet18)
    model=models.resnet18(pretrained=True)
    model.eval() # freeze BatchNorm and Dropout  
    
    full_weight_quantization_model(
       model,
       dtype=torch.float16, fp8_format='FP8_E4M3')
    