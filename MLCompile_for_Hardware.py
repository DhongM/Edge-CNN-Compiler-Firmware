
import torch
import torchvision.models as models
import torch.nn as nn
from ThisBN_Conv_Fused import completed_fusedModel_for_quantization 
from weight_engine import  full_weight_quantization_model
from InputActivation_Scale_Zp import calibrate_network,get_inputimage
import torch.fx
from dynamic_mapper import dynamic_topology_mapper
from bias_engine import compute_hardware_biases
from binary_exporter import export_model_bin
from eval_engine import verify_full_model_accuracy

def compile_entir_network(raw_model: nn.Module, inputimage_dir: str,
                          dataset_mean:list, dataset_std:list, batch_size: int = 32,
                          Target_Precision: torch.dtype=torch.float16,
                          fp8_format: str='FP8_E4M3'):
    """
    Universal ML Compiler. Takes ANY PyTorch CNN, calibrates it, 
    quantizes it, and exports it to a hardware-ready binary file.
    """
    model_name = raw_model.__class__.__name__
    print(f"🚀 Booting Master AI Hardware Compiler for {model_name}: ")
    
    # String mapping for the functions that require string inputs
    string_precision = "bf16" if Target_Precision== torch.bfloat16 else "fp16"

    # =========================================================
    # PHASE 1: Load and Fuse
    # =========================================================
    print(f"\n[Phase 1] Fusing BatchNorms and Casting Precision for {model_name}...")
    # We pass the raw model in, and get the hardware-ready model out!
    raw_model.eval()
    fused_model = completed_fusedModel_for_quantization(
        model=raw_model, 
        dtype=Target_Precision
    )
    orig_model_dtype=next(raw_model.parameters()).dtype
    print(f" checking orig_model dtype: { orig_model_dtype } ")
    
    fused_model_dtype = next(fused_model.parameters()).dtype
    print(f" checking fused model dtype: {fused_model_dtype } ")

    # =========================================================
    # PHASE 2: Extract Weight Quantization 
    # =========================================================
    print("\n[Phase 2] Extracting and Quantizing Weights...")
    Hardware_Payload={}
    Hardware_Payload=full_weight_quantization_model(fused_model, dtype=Target_Precision,fp8_format=fp8_format) 
    
    # =========================================================
    # PHASE 3: Input Quantized Scale and Zero Engine
    # =========================================================
    print("\n[Phase 3] Extracting Quantized Scale and Zero Point of Input...")
    
    calibration_dict=calibrate_network(model=fused_model, inputimage_dir=inputimage_dir, 
                     dataset_mean=dataset_mean, dataset_std=dataset_std, batch_size= batch_size,
                      num_calibration_batches=8, Intbitwidth= 8, strategy= 'percentile',
                      pct=99.9, precision=string_precision, num_bins=2048, device= 'cpu', fp8_format=fp8_format)
    # =========================================================
    # PHASE 4: Topology Map & INT32 Biases
    # =========================================================
    hardware_payload_update, mapping=dynamic_topology_mapper(model=fused_model, hardware_payload=Hardware_Payload,
                    calib_dict=calibration_dict) 
    
    # Compute INT32 Biases
    hardware_payload_final=compute_hardware_biases(hardware_payload_update)

    # =========================================================
    # PHASE 5: Binary Export & Metrics
    # =========================================================
    print("\n[Phase 5] Exporting to Hardware.bin...")

    ExportSummary= export_model_bin(hardware_payload=hardware_payload_final, output_dir = './hardware_bins', 
                     model_name=model_name)
    
    # ---------------------------------------------------------
    #  Calculate and Compare Model Sizes
    # ---------------------------------------------------------
    
    # Get total number of parameters in the model
    total_params = sum(p.numel() for p in raw_model.parameters())
    orig_fp32_size_mb = (total_params * 4) / (1024 * 1024)
    orig_fp16_size_mb = (total_params * 2) / (1024 * 1024)
    
    # Get New Hardware Binary Size in MB
    new_size_mb = ExportSummary.total_bytes / (1024 * 1024)
    
    # Calculate Compression against the FP32 Industry Standard
    compression_ratio_fp32 = (1 - (new_size_mb / orig_fp32_size_mb)) * 100

    print("\n" + "=" * 60)
    print(" 📊 COMPILER COMPRESSION REPORT (INDUSTRY STANDARD)")
    print("=" * 60)
    print(f"  Standard FP32 Baseline     : {orig_fp32_size_mb:.2f} MB")
    print(f"  Current FP16 Model Size    : {orig_fp16_size_mb:.2f} MB")
    print(f"  Compiled INT8/FP8 Firmware : {new_size_mb:.2f} MB")
    print("-" * 60)
    print(f"  Total Compression Achieved : {compression_ratio_fp32:.2f}% Reduction (vs FP32)!")
    print("=" * 60)


    # ---------------------------------------------------------
    # NEW: Phase 6 - Final INT8/FP8 Accuracy Parity Check
    # ---------------------------------------------------------
    _, dataloader = get_inputimage(
        inputimage_dir=inputimage_dir, 
        dataset_mean=dataset_mean, 
        dataset_std=dataset_std, 
        batch_size=batch_size
    )
    
    # Pass the hardware payload so it can build the Fake Quantization simulator!
    verify_full_model_accuracy(raw_model, fused_model, hardware_payload_final, dataloader)

    return hardware_payload_final, calibration_dict, ExportSummary

    


if __name__ == "__main__":
    
    # Shared Dataset Config
    image_dir="./datasets/imagenette2-160/val"
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std  = [0.229, 0.224, 0.225]

    # ---------------------------------------------------------
    # TEST 1: Compile ResNet18
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" COMPILING RESNET-18")
    print("="*60)
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
   

    compile_entir_network(raw_model=resnet, inputimage_dir=image_dir,
                          dataset_mean=dataset_mean, dataset_std=dataset_std, batch_size=32,
                          Target_Precision=torch.float16,
                          fp8_format='FP8_E4M3')

    
    