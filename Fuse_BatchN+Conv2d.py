import torch
import torch.nn as nn
import copy
from typing import Optional, List, Tuple
import torchvision.models as models


# =============================================================
# SECTION 1: CORE MATH — Fuse Single Conv2d + BatchNorm2d
# =============================================================

def fuse_conv_bn_math(
    conv:nn.Conv2d,
    bn:nn.BatchNorm2d)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes fused W_fused and B_fused from Conv2d + BatchNorm2d.

    Math:
        scale_factor[c] = bn.weight[c] / sqrt(bn.running_var[c] + bn.eps)
        W_fused[c]      = conv.weight[c] * scale_factor[c]
        B_fused[c]      = (b_conv[c] - bn.running_mean[c])
                          * scale_factor[c] + bn.bias[c]

    Args:
        conv : nn.Conv2d   (may or may not have bias)
        bn   : nn.BatchNorm2d
        b_conv[c]: bias of conv2d
        bn.bias[c]: bias of BatchNorm2d
        
    Returns:
        W_fused : fused weight tensor, same shape as conv.weight
        B_fused : fused bias tensor,   shape [out_channels]
      
    Notes:
        - All computation in float32 for numerical stability
        - Input conv/bn can be fp16/bf16 — promoted to float32
        - Output W_fused/B_fused returned in original dtype
    """    
   
    # ── Promote to float32 for stable computation ──────────────
    # BN running stats are always float32 in PyTorch
    # Conv weight may be fp16/bf16 — promote for math
    
    orginal_dtype=conv.weight.dtype
    weight=conv.weight.float()   # [out_ch, in_ch, kH, kW]
    bias_conv=conv.bias.float()\
              if conv.bias is not None\
              else torch.zeros(conv.out_channels, dtype=torch.float32)
              
    # BatchNorm parameters — always float32
    bn_weight =bn.weight.float()   # BatchNormal weight [out_ch]  scale
    β =bn.bias.float()    # BatchNormal bias, [out_ch]  shift
    μ =bn.running_mean.float()   # [out_ch]  mean
    bn_var=bn.running_var.float()  # [out_ch]  variance
    ε =bn.eps   # scalar    epsilon
    
    # ── Compute scale factor per output channel ────────────────
    # scale_factor[c] = γ[c] / sqrt(σ²[c] + ε) 
    scale_factor=bn_weight/torch.sqrt(bn_var+ε)  # [out_ch]
    
    # ── Compute W_fused ───────────────────────────────────────
    # W_fused[c] = W[c] * scale_factor[c]
    # W shape: [out_ch, in_ch, kH, kW]
    # scale_factor shape: [out_ch] → reshape for broadcast
    # → [out_ch, 1, 1, 1]
    scale_factor_reshap=scale_factor.view(-1,1,1,1) # broadcast
    Weight_fused= weight*scale_factor_reshap
    
    # ── Compute B_fused ───────────────────────────────────────
    # B_fused[c] = (b_conv[c] - μ[c]) * scale_factor[c] + β[c]
    Bias_fused=(bias_conv-μ)*scale_factor+β
    
    # ── Return in original dtype ──────────────────────────────
    # Conversion back to fp16/bf16 done here
    # Weight quantization will then quantize W_fused → INT8

    return Weight_fused.to(orginal_dtype), Bias_fused.to(orginal_dtype)
   

# =============================================================
# SECTION 2: REPLACE Conv2d with Fused Weight and Bias
# =============================================================

def build_fused_conv(conv:nn.Conv2d,
     bn: nn.BatchNorm2d)->nn.Conv2d:
         
    """
    Creates a new nn.Conv2d with fused weights and bias.
    BatchNorm is completely absorbed — no BatchNorm on accelerator.

    Args:
        conv : original nn.Conv2d
        bn   : following nn.BatchNorm2d

    Returns:
        fused_conv : nn.Conv2d with W_fused and B_fused
                     same hyperparameters as original conv
                     always has bias=True (B_fused always exists)
    """    
    Weight_fused, Bias_fused=fuse_conv_bn_math(conv,bn)
    fused_conv=nn.Conv2d(out_channels=conv.out_channels,
               in_channels=conv.in_channels,
               kernel_size=conv.kernel_size,
               stride=conv.stride,
               padding=conv.padding,
               dilation=conv.dilation,
               groups=conv.groups,      
               bias=True,   # always True after fusion
               padding_mode=conv.padding_mode)# preserve original dtype   
               
    # update weight and bias with fused parameters
    fused_conv.weight.data=Weight_fused
    #print(f" weight fused data type {Weight_fused.dtype}")
    fused_conv.bias.data=Bias_fused
    fused_conv=fused_conv.to(Weight_fused.dtype)  
    
    return fused_conv

# =============================================================
# SECTION 3: MODEL FUSION
# =============================================================

def fuse_module_recursive(parent:nn.Module,prefix: str = ""):
    """
    Recursively walks model tree.
    Finds Conv2d immediately followed by BatchNorm2d.
    Replaces pair with fused Conv2d + Identity.

    Uses named_children() to iterate one level at a time.
    Recursion handles nested modules (Sequential, ResNet blocks).
    Args:
        parent: The current module block
        prefix: Tracks the full global name path (e.g., "layer1.0.")
    """
    # Get list of child names and modules
    children=list(parent.named_children())
     
    i=0
    while i<len(children):
        name_i, module_i=children[i]
        # Check if this is Conv2d followed by BatchNorm2d
        full_name_i = prefix + name_i
        if i+1<len(children):
            name_j,module_j=children[i+1]
            full_name_j = prefix + name_j
           
            if isinstance(module_i,nn.Conv2d) and isinstance(module_j,nn.BatchNorm2d):
              # ── Found Conv+BN pair — fuse them ────────────                  
                assert module_i.out_channels==module_j.num_features,\
                    (f"Mismatch:Conv out_channels:"
                    f"{module_i.out_channels}!="
                    f" BatchNorm num_features:{module_j.num_features}")

                # Build fused Conv2d
                fused_conv_bn=build_fused_conv(module_i, module_j)
              
                # Replace Conv2d with fused version
                setattr(parent, name_i, fused_conv_bn)
              
                # Replace BatchNorm2d with Identity
                # Identity: output = input (pass-through)
                # Accelerator never sees BatchNorm 
                setattr(parent,name_j, nn.Identity())

                # --- BEAUTIFUL TERMINAL LOGGING ---
                print(f" [Fused] {full_name_i} + {full_name_j}")
                print(f"    ├─ Weigh_fused : {tuple(fused_conv_bn.weight.shape)}")
                print(f"    ├─ Bias_fused : {tuple(fused_conv_bn.bias.shape)}")
                print(f"    └─ {full_name_j} -> Replaced by nn.Identity()")
                
                i+=2  # skip both Conv and BN
                continue              
                
        # Not a Conv+BatchNorm pair — recurse into children 
        if len(list(module_i.children()))>0:
           # Pass the current full name down as the new prefix (e.g., "layer1.0.")
           fuse_module_recursive(module_i,prefix=full_name_i+".")    
        i+=1


def fused_model(model:nn.Module,inplace,bool=False)->nn.Module:
    """
    Fuses ALL Conv2d + BatchNorm2d pairs in model.
    Searches entire model recursively for Conv→BN patterns.

    IMPORTANT: Must be called BEFORE:
      - Weight quantization (quantize_weights_per_channel)
      - Activation calibration (calibrate_layers_2_to_32)
      - Any PCIe transfer

    Must be called AFTER:
      - model.eval()  ← BN uses running stats not batch stats
      - Training complete ← running_mean/var must be final
    Args:
        model   : CNN model in eval mode
        inplace : False → returns new model (safe)
                  True  → modifies model in place (faster)
    Returns:
        fused_model : model with all Conv+BN pairs fused
                      BN layers replaced with nn.Identity()
    """
    if not inplace:
       # Deep copy — original model unchanged 
       model=copy.deepcopy(model) 
    
    # Find and fuse all Conv+BN pairs
    # Must use named_children() for in-place replacement  
    fuse_module_recursive(model)  
    return model

# =============================================================
# SECTION 4: VERIFICATION
# =============================================================
def verify_fusion_accuracy (original_model:nn.Module,
           fused_model: nn.Module,
           input_shape: Tuple=(4,3,224,224)
           )->bool:
    """
    Verifies fused model produces same output as original.
    Runs random input through both models and compares.
    Args:
        original_model : unfused model
        fused_model    : fused model
        input_shape    : test input shape [N, C, H, W]
        dtype          : test dtype
        tolerance      : max allowed difference
    Returns:
        True if outputs match within tolerance
    """
    original_model.eval()
    fused_model.eval()
    # Random test input
    first_param = next(original_model.parameters())
    device=first_param.device
    dtype=first_param.dtype
    # 2. Create the random image using the EXACT same dtype and device
    x=torch.randn(input_shape, dtype=dtype, device=device) 

    tolerance_map = {
        torch.float32:  1e-5,
        torch.float16:  2e-2,   # ← was 1e-4, now correct for fp16
        torch.bfloat16: 9e-2,   # ← was 1e-4, now correct for bf16
    }
    tolerance = tolerance_map.get(dtype, 1e-3)
    

    with torch.no_grad():
         output_original=original_model(x).float()
         output_fused=fused_model(x).float()         

    # Compare outputs
    max_diff=(output_original-output_fused).abs().max().item()
    mean_diff=(output_original-output_fused).abs().mean().item()
    match=max_diff <tolerance
    
    print(f"\nFusion Verification:")
    print(f"  Weight dtype    : {dtype}")
    print(f"  Max  difference : {max_diff:.2e}  "
          f"{'Good' if match else ' FAIL'}")
    print(f"  Mean difference : {mean_diff:.2e}")
    print(f"  Tolerance       : {tolerance:.2e}")
    print(f"  Result          : "
          f"{'PASSED' if match else ' FAILED'}")
    
    return match
    
 
def count_module(model:nn.Module)->dict:
    """
    Count Conv2d, BatchNorm2d, Identity layers.
    Correct: Shows fusion reduced BN count to zero.
    """
    counts={
         'Conv2d':0,
         'BatchNorm2d':0,
         'Identity':0,
         'Linear':0,
         'ReLU':0}

    for module in model.modules():
        name=type(module).__name__
        if name in counts:
            counts[name]+=1
    return counts
 
 
# =============================================================
# SECTION 5: COMPLETE PIPELINE
# =============================================================
def completed_fusedModel_for_quantization(
       model:nn.Module,
       dtype: torch.dtype=torch.float16)->nn.Module:
    """
    Complete model preparation before quantization:
      Step 1: Set eval mode      ← BN uses running stats
      Step 2: Convert to dtype   ← fp16 or bf16
      Step 3: Fuse Conv+BN       ← this function
      Step 4: Verify fusion      ← check correctness

    Returns fused model ready for:
      → quantize_weights_per_channel()
      → calibrate_layers_2_to_32()

    Args:
        model : trained CNN model
        dtype : fp16 or bf16 for deployment
    Returns:
        fused_model : ready for weight quantization and layer2 above quantization
    """    
    print("=" * 60)
    print("  CNN Model Preparation for Quantization")
    print("=" * 60)
  
    # ── Step 1: Set eval mode ──────────────────────────────────
    # CRITICAL: must be eval before fusion
    # Training mode: BN uses batch statistics (wrong for fusion)
    # Eval mode:     BN uses running_mean/var (correct) 
    print("\nStep 1: Set eval mode")
    model.eval()
    print("  model.eval()  BrachNorm2d now uses running_mean/var")

    # ── Step 2: Convert dtype ──────────────────────────────────
    print(f"\nStep 2: Convert to {dtype}")
    
    original_model = model.to(dtype)
    print(f"  Model data converted to {dtype} ")
    
    # Count before fusion
    counts_before_fused= count_module(original_model)
    print(f"  Count before fusion:")
    for k,v in counts_before_fused.items():
        if v>0:
            print(f" key {k:<15}: {v}")
    
    # ── Step 3: Fuse Conv+BrachNorm2d ───────────────────────────────────
    print(f"\nStep 3: Fuse Conv2d + BatchNorm2d")
    
    print(f"  Starting fusion:")
    # Run fusion
    model_fused=fused_model(original_model, inplace=False)
    model_fused=model_fused.to(dtype)

    # Count after fusion
    counts_after_fused= count_module(model_fused)
    print(f"  Count after fusion:")
    for k,v in counts_after_fused.items():
        if v>0:
            print(f" key {k:<15}: {v}")
            
    # Verify BatchNorm2d eliminated        
    batchNorm_remaining = counts_after_fused['BatchNorm2d'] 
    identity_added     = counts_after_fused['Identity']
    
    print(f"\n  BatchNorm2d remaining : {batchNorm_remaining}  "
          f"{'Correct' if batchNorm_remaining == 0 else 'Wrong'}")
    print(f"  Identity layers added : {identity_added}  "
          f"(replaced batchNorm  positions)")

    # ── Step 4: Verify accuracy ───────────────

    verify_fusion_accuracy(original_model=original_model,
           fused_model=model_fused,
           input_shape=(4,3,224,224)
           )

    print(f"\n{'='*60}")
    print(f"  Fused model ready for weight and layer>2 quantization")
    print(f"  Next steps:")
    print(f"    1. quantize_weights_per_channel(fused_model)")
    print(f"    2. calibrate_layer1(data_dir images) for input activation scale in layer 1")
    print(f"    3. calibrate_layers_2_to_32(fused_model, data_dir) for input activation scale in layer 2 above")
    print(f"    4. write_calibration_bin(...)")
    print(f"{'='*60}")
    
    return model_fused


# =============================================================
# SECTION 6: DEMO
# =============================================================

if __name__ == "__main__":

    print("BatchNorm Fusion Demo — ResNet18")
    print("=" * 60)

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Run complete preparation pipeline
    fusedmodel = completed_fusedModel_for_quantization(
        model = model,
        dtype = torch.float16
    )

    # Show example fused layer parameters
    print("\nExample — First fused Conv2d:")
    first_conv = None
    for name, module in fusedmodel.named_modules():
        if isinstance(module, nn.Conv2d) \
           and module.bias is not None:
            first_conv = (name, module)
            break

    if first_conv:
        name, conv = first_conv
        print(f"  Layer      : {name}")
        print(f"  Weight_fused    : {tuple(conv.weight.shape)}  "
              f"{conv.weight.dtype}")
        print(f"  Bias_fused    : {tuple(conv.bias.shape)}  "
              f"{conv.bias.dtype}")
        print(f"  Weight range    : [{conv.weight.min().item():.4f}, "
              f"{conv.weight.max().item():.4f}]")
        print(f"  Bias range    : [{conv.bias.min().item():.4f}, "
              f"{conv.bias.max().item():.4f}]")
        print(f"\n  Ready for quantize_weights_per_channel() ✅")