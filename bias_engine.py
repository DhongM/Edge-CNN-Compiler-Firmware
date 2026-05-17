import torch
import torchvision.models as models
import torch.nn as nn

def compute_hardware_biases(hardware_payload: dict) -> dict:
    """
    Computes the final INT32 Hardware Biases including Zero-Point Unfolding.
    Formula: Final_Bias = Round(Bias_FP16 / (S_w * S_in)) - (Z_in * sum(W_int8))
    Quantization bias except FC layer, Bias in FC layer should keep FP16/BF16
    """
    print("\n" + "=" * 60)
    print(" 🧠 [Bias Engine] Computing INT32 Biases & Zero-Point Offsets for layers excpet FC layer")
    print("=" * 60)
    INT32_MIN, INT32_MAX = -2147483648, 2147483647
    QUANT_BLACKLIST=['fc','classifier','head'] # Layers to keep in FP8
    for layer_name, data in hardware_payload.items():
        if  data.get("type")!="FP8_E4M3" and data.get("type")!="FP8_E5M2":
            # 1. Gather all required data
            w_int8=data["quantizedweights_channel"]     # [out_ch, in_ch, k, k] or [out_ch, in_ch]
            w_scale=data["weight_scale"]                # [out_ch], scale per channel, fp16 or bf16
            orginal_dtype=w_scale.dtype               
            i_scale=data["input_scale"]                 # Scalar
            z_input = data["input_zero_point"]          # Scalar (Integer)
            out_channels=w_int8.shape[0]
            # -----------------------------------------------------------------
            # PART A: The Base INT32 Bias Calculation
            # -----------------------------------------------------------------
            
            if data.get("bias") is not None: 
                orginal_bias=data["bias"]            #should be fp16 or bf16
                # 1. convert to fp32 for calculating 
                orginal_bias= orginal_bias.float()   #convert to fp32 for calcuation    
                bias_scale_raw=w_scale.float()*i_scale.float()
                bias_scale_raw=bias_scale_raw.clamp(min=1e-8)  # Guard #should be fp32
                print(f"bias_scale raw type: {bias_scale_raw.dtype}")
                
                # 2. quantize bias, Qbias = round(bias / S_bias) 
                bias_base_int32=torch.round(orginal_bias/bias_scale_raw)
                print(f"bias type during calcuating should be float32:{bias_base_int32.dtype}")
                bias_base_int32=bias_base_int32.clamp(INT32_MIN, INT32_MAX).to(torch.int32)

                bias_scale=bias_scale_raw.to(orginal_dtype)
                print(f"bias_scale type:{bias_scale.dtype}")
                hardware_payload[layer_name]["bias_scale"] = bias_scale
            else:
                bias_base_int32=torch.zeros(out_channels, dtype=torch.int32) 
            # ----------------------------------------------------------------- 
            # PART B: Zero-Point Unfolding (The Cross-Term Offset)
            # -----------------------------------------------------------------
            # For Conv2d: sum across dim 1, 2, 3. For Linear: sum across dim 1.
            # Round(Bias_FP16 / (S_w * S_in))- (Z_in * sum(W_int8))
            if z_input!=0:
               dims_to_sum=list(range(1, w_int8.dim()))   # [1,2,3] for Conv2d
               w_int32=w_int8.to(torch.int32)           
               weight_sum=w_int32.sum(dim=dims_to_sum)   # [out_ch]
               # Offset = Z_input * Sum(W_int8)
               zp_offset=z_input*weight_sum
            else:
               zp_offset=torch.zeros(out_channels, dtype=torch.int32)
            # -----------------------------------------------------------------
            # PART C: The Final Fold
            # -----------------------------------------------------------------
            # Hardware Bias = Base_Bias - Offset
            final_bias=bias_base_int32-zp_offset

            # Clamp to physical 32-bit integer limits to prevent register overflow
            final_bias_int32=final_bias.clamp(INT32_MIN, INT32_MAX).to(torch.int32)
            
            hardware_payload[layer_name]["bias"]=final_bias_int32
             # Print verification
            had_original="yes" if data.get("bias") is not None else "No"
            print(f"  {layer_name:<18} | Orig Bias: {had_original:<3}\n"
                  f"    ├─ Zero_Point of input: {z_input:<4}  | Created INT32_Bias: {tuple(final_bias_int32.shape)}\n"
                  f"    ├─ INT32_Bias min: {final_bias_int32.min().item()}, INT32_Bias max: {final_bias_int32.max().item()}\n"
                  f"    ├─ Bias type : {hardware_payload[layer_name]['bias'].dtype}\n"
                  f"    ├─ Bias min: {hardware_payload[layer_name]['bias'].min().item()}\n" 
                  f"    ├─ Bias max: {hardware_payload[layer_name]['bias'].max().item()}\n"
                  f"    ├─ bias scale dtype { hardware_payload[layer_name]['bias_scale'].dtype}")
        else:
            if  any(top_name in layer_name for top_name in QUANT_BLACKLIST ):
                if data.get("type")=="FP8_E4M3" or data.get("type")=="FP8_E5M2":
                   print(f"  {layer_name:<18} | FP8 Path: Retaining pure FP16/BF16 bias.")
                   print(f"    ├─ Bias min: {hardware_payload[layer_name]['bias'].min().item()}\n"
                         f"    ├─ Bias max: {hardware_payload[layer_name]['bias'].max().item()}")
                        
         
    print("=" * 60) 
    return hardware_payload