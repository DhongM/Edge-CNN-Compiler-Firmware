import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Tuple
#import torch.ao.quantization.intrinsic as nni
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization import prepare, convert
import torchvision.models as models


def quant_max_min(Intbitwidth:int):
    quant_max=(1<<(Intbitwidth-1))-1
    quant_min=-(1<<(Intbitwidth-1))
    return quant_max, quant_min
    
def tensor_scale_zero_Asymmetric(fp_min:float,fp_max:float, Intbitwidth:int):
   
    """Asymmetric scale and zero_point computation."""
    
    quan_max, quan_min=quant_max_min(Intbitwidth)
    
    #scale float point, zero point is integer
    scale=max((fp_max-fp_min)/float(quan_max-quan_min),1e-5)
    
    #zero_point=quan_min-fp_min/scale
    
    zero_point=round(-fp_min/scale+quan_min)
    
    if(zero_point>quan_max):
       zero_point=quan_max
    elif zero_point<quan_min:
       zero_point=quan_min
    else:
       zero_point=int(zero_point)
    return scale, zero_point

""" offline static input_scale and zero_point of input activaiton"""  
def get_inputimage(
    inputimage_dir: str, 
    dataset_mean: list, 
    dataset_std: list, 
    batch_size: int
) -> Tuple[torch.Tensor, DataLoader]:
    """
    ╔══════════════════════════════════════════════════╗
    ║  SHARED — used by Layer 1 AND Layer 2-32        ║
    ║                                                  ║
    ║  Layer 1:    images ARE the input — used        ║
    ║              DIRECTLY for scale computation      ║
    ║                                                  ║
    ║  Layer 2 above: images fed to model forward pass  ║
    ║              hooks capture intermediate acts    ║
    ║                                                  ║
    ║  Both cases need the SAME calibration images   ║
    ║  with SAME production transforms               ║
    ╚══════════════════════════════════════════════════╝
    """ 
    # Production transforms — MUST match inference pipeline exactly
    production_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    
    try:
        calib_dataset = datasets.ImageFolder(root=inputimage_dir, transform=production_transforms)
        calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=True)
        
    except FileNotFoundError:
        error_msg = (
            f"\n[WARNING] Folder '{inputimage_dir}' not found.\n" 
            f"Please check your dataset path, download real images, and try compiling again."
        )
        raise RuntimeError(error_msg)
        
    # Grab one batch to verify
    images_tensor, _ = next(iter(calib_loader))
    
    
    print(f"Loaded {images_tensor.shape[0]} real images per batch from {inputimage_dir}")
    
    return images_tensor.float(), calib_loader
        


def input_activation_minmax(
    data_source, # torch.Tensor or DataLoader
    strategy: str='percentile', 
    pct: float=99.9,
    num_bins: int=2048,
    precision: str= "fp16"
    )->Tuple[float,float]:
        
    """
    Automatically selects method based on input type:
      torch.Tensor  → exact torch.quantile (fast, small data)
      DataLoader    → histogram CDF        (memory efficient)
    SHARED: called by Layer 1 AND Layer 2-32 calibration
    precision is user choice
    precision is fp16, convert data_source to fp16 before min and max
    precision is bf16, covert data_source to bf16 before min and max
    HARDWARE SIMULATION: Casts to target precision to mimic PCIe bus truncation,
    then computes stats in FP32 to prevent zero-point amplification errors.
    """
    # --- HARDWARE SIMULATION HELPER ---
    def simulate_hardware(tensor: torch.Tensor, prec: str) -> torch.Tensor:
        if prec == "bf16":
            return tensor.to(torch.bfloat16).float()
        elif prec == "fp16":
            return tensor.to(torch.float16).float()
        return tensor.float()

       # PATH A: Tensor path
    if isinstance(data_source, torch.Tensor): 
        data_source_flat  = simulate_hardware(data_source, precision).flatten() 
        
        max_elements = 16_000_000
        if data_source_flat.numel() > max_elements:
            # Calculate how many elements to skip to get under 16M
            stride = (data_source_flat.numel() // max_elements) + 1
            # Subsample the array (e.g., [::10] takes every 10th element)
            data_source_flat = data_source_flat[::stride]
        if strategy == 'minmax':
            return data_source_flat.min().item(), data_source_flat.max().item()

        elif strategy == 'percentile':
            lower = (100.0 - pct) / 100.0
            upper = pct / 100.0
            return (
                torch.quantile(data_source_flat, lower).item(),
                torch.quantile(data_source_flat, upper).item()
                )
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'")
                      
    # =========================================================================

    # PATH B: THE DATALOADER PATH (2-Pass Histogram, OOM-Proof)

    # =========================================================================
    elif isinstance(data_source, DataLoader): 

        # FIX #2: Warn the engineer if they accidentally left Data Augmentation on

        if hasattr(data_source.dataset, 'transform') and "Random" in str(data_source.dataset.transform):

            print("[WARNING] Random transforms detected in DataLoader!")
            print("Calibration requires static data. Scales may suffer slight variance.")         
        
        percentile_decimal = pct / 100.0      
        
        #Pass 1: Measure the Room (Find Absolute Bounds)
        abs_min=float('inf')
        abs_max=float('-inf')
        
        for batch in data_source:
            # Handle DataLoaders that return (image, label) tuples
            if isinstance(batch, (list, tuple)):
               batch=batch[0]
               
            # Simulate hardware on this specific batch!   
            batch_sim=simulate_hardware(batch, precision)
               
            abs_min=min(abs_min, batch_sim.min().item())
            abs_max=max(abs_max, batch_sim.max().item())
        
       
        # Safety catch: If all data is exactly zero
        if abs_min == abs_max:
        #   return abs_min, abs_max+ 1e-5
            abs_max += 1e-5
        
        # ---------------------------------------------------------
        # Pass 2: Build the Shelves (Histogram Tally)
        # ---------------------------------------------------------
        hist=torch.zeros(num_bins)
        for batch in data_source:
            if isinstance(batch, (list, tuple)): 
                batch = batch[0]
            # Simulate hardware on this specific batch!   
            batch_sim=simulate_hardware(batch, precision)
            
            hist+=torch.histc(batch_sim,
            bins=num_bins,
            min=abs_min,
            max=abs_max)
    
       # Pass 3: Calculate CDF Percentile
        hist/=hist.sum()
        cdf=hist.cumsum(dim=0)
        bin_w=(abs_max-abs_min)/num_bins
        
        # Find the bucket indices where our percentile limits are crossed
        lo_idx=(cdf>=(1.0-percentile_decimal)).nonzero()[0].item()
        hi_idx=(cdf>=percentile_decimal).nonzero()[0].item()
        
        return(abs_min+(lo_idx * bin_w),abs_min+(hi_idx*bin_w))
        
    else:
        raise TypeError(f"[critical error] invalid data_source type for finding min and max.\n"
                        f" Expected:'torch.Tensor' or 'torch.utils.data.Dataloader'\n"
                        f"received''{type(data_source)}'\n")
                        
        

def calibrate_layer1(
    inputimage_dir:    str,
    dataset_mean:list, 
    dataset_std:list,
    batch_size:  int   = 32,
    Intbitwidth: int   = 8,
    strategy:    str   = 'percentile',
    pct:         float = 99.9,
    precision:str="fp16",
    num_bins: int=2048
     ) -> dict:
    """
    ╔═════════════════════════════════════════════════╔ 
    ║  LAYER 1 SPECIFIC CALIBRATION                   ║
    ║                                                 ║
    ║  Complete pipeline for Layer 1 only:            ║
    ║    1. Load images                               ║
    ║    2. Collect min and max                       ║
    ║    3. Compute scale+zp                          ║
    ║    4. Return LayerQuantParams                   ║
    ╚═════════════════════════════════════════════════╚

    Args:
        inputimage_dir    : path to  image folder
        batch_size  : number of calibration images
        Intbitwidth : target bitwidth (default 8)
        strategy    : 'minmax', 'percentile', 
        pct         : percentile value
        dataset_mean: image dataset mean
        dataset_std : image dataset variance
        precision   : user impage type or precsion FP16/BF16
    Returns:
        LayerQuantParams with input_images with FP16/BF16 (depending on useer) and 
        with QS_Input[1] and ZP_Input[1]
        ready for PCIe serialization
    """


    print("\n" + "=" * 60)
    print("  CNN Layer 1 Static Offline Input Image Calibration")
    print("=" * 60)
   
    # ── Step 1: Load images from hard drive downloaded from imageNet───────────────────────
    # Layer 1: images used DIRECTLY as activation tensor. 
    # Layer 2-32: images fed to model, hooks capture acts
    # images to be send to AI accelerator 
    print("\nStep 1: Load calibration images [SHARED]")

    calibration_images,full_dataloader =get_inputimage(inputimage_dir, dataset_mean, dataset_std, batch_size=batch_size)

    # ── Step 2: Collect Layer 1 min and max  ─────────
    # LAYER 1 SPECIFIC: raw images ARE the activation
    # input_activation_minmax Simulate hardware datatype alreadly!
    fp_min, fp_max=input_activation_minmax(full_dataloader, strategy, pct, num_bins, precision)   

    # ── Step 3: Compute scale and zero_point ──────────────────
    # Layer 1 fp_min/fp_max: from raw images 
    print("\nStep 3: Compute scale + zero_point  [layer 1]")
    scale, zero_point =tensor_scale_zero_Asymmetric(fp_min,fp_max, Intbitwidth=8)

    print(f"Check Scale and Zero point dtype,scale type:{type(scale)}, zero point: {type(zero_point)}")
    # fp16 and bf16 scale will be sent to AI accelerator by PCIe!
    if precision=="fp16":
       scale_16 = torch.tensor(scale, dtype=torch.float16) 
    elif precision=="bf16" :
       scale_16 =torch.tensor(scale,dtype=torch.bfloat16)       
      
    print(f" Scale_Input[1] = {scale_16 .item():.8f}  ({precision} and  {scale_16.dtype})")
    print(f" ZP_Input[1] = {zero_point}  (int8) and  {type(zero_point)}")
   
    # ── Step 4: convert images to bf16/fp16 ──────────────────
    if precision=="bf16":
      calibration_images= calibration_images.to(torch.bfloat16)
    elif precision=="fp16":
      calibration_images= calibration_images.to(torch.float16)
     
       
    return {"verification_images": calibration_images,
             "layer_name":   "layer1_input",
             "layer_index": 1,
             "scale_layer1": scale_16, 
             "zero_point":zero_point }
 

class LayerHooker:
    """
    LAYER 2-32 SPECIFIC.
    Manages forward hooks on all target modules.
    Key design decisions:
      - Hooks capture OUTPUT of each module
        Output of Layer N = Input of Layer N+1
        → This gives us QS_Input for layer N+1
    """

    def __init__(self, model:nn.Module):
        self.model=model
        self.captured_activations={}
        self.hook_handles=[]
        # Layer index — tracks which layer index each name is
        self.layer_index: list[str] = []
  
    def _get_hook(self, layer_name: str):
        """
        Creates forward hook closure for one layer.
        The actual 'Wiretap'-hook that intercepts the data
        Called automatically during forward pass.
        """ 
        def hook(model, input, output):
            # 1. Initialize the list for this layer if it doesn't exist
            if layer_name not in self.captured_activations:
               self.captured_activations[layer_name] = [] 
                
            # 2. THE MEMORY SECRET: .detach().cpu()
           
            self.captured_activations[layer_name].append(output.detach().cpu())
        return hook
 
    def register_hooks(self, 
          target_modules:tuple=(nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,nn.Linear),
          skip_names: list[str]=None):
        """
        LAYER 2-32 SPECIFIC.
        Register forward hooks on target module types.
        WHY hook on ReLU/activation functions:
          ReLU output = Conv2d output after activation
                      = input to NEXT Conv2d layer
          → Directly gives us the activation distribution
            that the next layer's input will have
          → This is QS_Input for next layer ✅

        WHY NOT hook on Conv2d output directly:
          Conv2d output is BEFORE BatchNorm and ReLU
          The actual input to next layer is AFTER BN+ReLU
          → Hooking Conv2d output gives wrong distribution
        Args:
            skip_names     : layer name substrings to skip
        """
        skip_names = skip_names or []
        hook_count = 0
        for name, module in self.model.named_modules():
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(module, target_modules):
               self.layer_index.append(name)
               
               handle=module.register_forward_hook(self._get_hook(name))
               self.hook_handles.append(handle)
               hook_count += 1
               print(f" [hooked] layer name is {name}")
        print(f"Registered {hook_count} total hooks.")
     
    def remove_hooks (self):
        for handle in  self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        print(f" removed all of hooks") 
 
 
 # SECTION : LAYER >=2 SPECIFIC — Forward Pass
def compute_entirenetwork_params(model:nn.Module,dataloader:DataLoader,
    num_calibration_batches: int=8, strategy:str='percentile', precision: str="bf16", device: str= 'cpu',fp8_format: str='FP8_E4M3'): 
        
        # Pre-Step: # dataloader has feed to model   ───────────────────────────────
        
        # ── Step 1: Register hooks ─────────────────────────────────
    # Prepare the Wiretaps and Forward Pass
    # LAYER 2 above SPECIFIC: not needed for Layer 1
    # Hooks on ReLU outputs = inputs to next Conv2d
    print("\nStep 1: Register hooks  [LAYER 2 above SPECIFIC]")
       
    calibrator=LayerHooker(model)
    calibrator.register_hooks(target_modules=(nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,nn.Linear),skip_names=None )
      
      
    # ── Step 2: Run forward pass ───────────────────────────────
    # Flow Data Through the Network (The "Forward Pass") LAYER 2 above SPECIFIC: not needed for Layer 1
    # Hooks fire automatically during forward pass
    # Each hook updates running stats in float32
    
    print("\nStep 2: Run calibration forward pass"
          "  [LAYER 2 above SPECIFIC]")
    
    print(f" Running {num_calibration_batches} batches through the network...")
    model.eval() # MUST be in eval mode
    model.to(device)
    
    print(f" Device: {device}")
    
   
    with torch.no_grad(): # Disable gradients to save massive amounts of RAM
        # Get the exact dtype of the model so we can match it
        model_dtype = next(model.parameters()).dtype
        print(f" checking model dtype: {model_dtype } ")

        for i, (images, labels) in enumerate(dataloader):
            if i >= num_calibration_batches:
                break # Stop after our subset (e.g., 8 batches of 32 = 256 images)
            # 1. Move images to correct device
            images=images.to(device)
            
            # 2. CAST IMAGES TO MATCH MODEL PRECISION (THIS FIXES THE CRASH)
            images = images.to(model_dtype)
            
            _ = model(images)
            
            if (i + 1) % 8 == 0:
                print(f"  Batch {i+1}/{num_calibration_batches} complete")
     
        total_batches = min(num_calibration_batches, i + 1)
        print(f" Forward pass complete: "
              f"{total_batches} batches processed")  
    
    # ── Step 3: Remove hooks ───────────────────────────────────
    # LAYER 2 above SPECIFIC: cleanup after calibration
    # MUST remove hooks to avoid memory leaks
    print("\nStep 3: Remove hooks  [LAYER 2 above SPECIFIC]")
    calibrator.remove_hooks()  
    
    if fp8_format == 'FP8_E4M3':
        fp8_max = 448.0
    elif fp8_format == 'FP8_E5M2':
        fp8_max = 57344.0
    else:
        raise ValueError(f"Unknown fp8_format: {fp8_format}")

     # ── Step 4: Compute scale + zero_point per layer ──────────
    # SHARED MATH: compute_scale_zeropoint() same as Layer 1
    # LAYER 2 above SPECIFIC: data comes from model not images
    
    print("\n" + "#" * 60)
    print("  Calculating Hardware Input Activation Scale and Zero Point Per Layer (LAYERS >=2)")
    print("=" * 60)
    
    hardware_parameters = {}
    layer_index=1
    #QUANT_BLACKLIST=['fc','classifier','head'] # Layers to keep in FP8
    for layer_name, tensor_list in calibrator.captured_activations.items():
        # Combine the 8 batches of intercepts into one giant Tensor
        # (This is why we only use ~256 images! Otherwise this tensor is too huge).
        merged_tensor=torch.cat(tensor_list, dim=0)

        # input_activation_minmax Simulate hardware datatype alreadly!
        fp_min,fp_max=input_activation_minmax(merged_tensor,strategy=strategy,
        pct=99.9,num_bins=2048, precision=precision)
        print(f" check fp_min and fp_max data type: fp_min:{fp_min:.6f}, fp_max:{type(fp_max)}")

         
        # ── 1. Always Calculate INT8 Asymmetric Math ───────────────
        scale_int8, zp_int8 = tensor_scale_zero_Asymmetric(fp_min, fp_max, Intbitwidth=8)
        # ── 2. Always Calculate FP8 Symmetric Math ─────────────────
        abs_max = max(abs(fp_min), abs(fp_max))
        abs_max = max(abs_max, 1e-5) # Guard against zero
        scale_fp8 = abs_max / fp8_max   
        zp_fp8 = 0

        # 5.Format the scale for the hardware .bin file   
        if precision=="bf16":
           scale_int8_hw = torch.tensor(scale_int8, dtype=torch.bfloat16)
           scale_fp8_hw  = torch.tensor(scale_fp8, dtype=torch.bfloat16)
           
        elif precision=="fp16":
           scale_int8_hw = torch.tensor(scale_int8, dtype=torch.float16)
           scale_fp8_hw  = torch.tensor(scale_fp8, dtype=torch.float16)
           
        layer_index+=1   
        # Save to dictionary
        hardware_parameters[layer_name]={
          "layer_index": layer_index,
          "scale_int8": scale_int8_hw,
          "zero_point_int8": zp_int8,
          "scale_fp8": scale_fp8_hw,
          "zero_point_fp8": zp_fp8
          }    
              
   
    return hardware_parameters
    
# SECTION 6: All of layers CALIBRATION FUNCTION
def calibrate_network(model:nn.Module, inputimage_dir: str, dataset_mean:list, 
    dataset_std:list, batch_size: int = 32,
    num_calibration_batches: int=8, Intbitwidth: int= 8, strategy:str = 'percentile',
    pct: float = 99.9, precision: str="bf16", num_bins: int=2048, device: str= 'cpu',fp8_format: str='FP8_E4M3'):
    
    # 1. Get the Data
    verification_images, dataloader_image = get_inputimage(
        inputimage_dir=inputimage_dir, dataset_mean=dataset_mean, 
        dataset_std=dataset_std, batch_size=batch_size
    )
    
    # 2. Calibrate Layer 1
    Layer1_parameter = calibrate_layer1(
        inputimage_dir=inputimage_dir, dataset_mean=dataset_mean, dataset_std=dataset_std, 
        batch_size=batch_size, Intbitwidth=Intbitwidth, strategy=strategy, pct=pct, 
        precision=precision, num_bins=num_bins
    ) 
  
    # ---------------------------------------------------------
    # Create the Master Dictionary and put Layer 1 FIRST
    # ---------------------------------------------------------
    hardware_parameters_allLayers = {}
    
    hardware_parameters_allLayers["layer1_input"] = {
        "layer_index":Layer1_parameter["layer_index"],
        "scale_int8": Layer1_parameter["scale_layer1"],
        "zero_point_int8": Layer1_parameter["zero_point"],
        "scale_fp8": Layer1_parameter["scale_layer1"],
        "zero_point_fp8": 0,
        "verification_images": Layer1_parameter["verification_images"]
    } 
    # 3. Calibrate Layers 2-32
    params_above_layer2 = compute_entirenetwork_params(
        model, dataloader=dataloader_image,
        num_calibration_batches=num_calibration_batches, strategy=strategy, precision=precision, 
        device=device,fp8_format=fp8_format)
    
    try:
        hardware_parameters_allLayers.update(params_above_layer2)
        print("\n" + "=" * 60)
        print(" FINAL HARDWARE EXPORT INPUT SUMMARY (ALL LAYERS)")
        print(f"{'Layer Name':<20} |{'Layer index':<7}| {'Scale':<15} | {'Zero Point'}")
        print("=" * 60)
            # Loop through the final combined dictionary and print everything beautifully
        for layer_name, params in  hardware_parameters_allLayers.items():
           layer_index=params["layer_index"]   
           scale_val_int8=params["scale_int8"] 
           scale_val_fp8=params["scale_fp8"] 
           zp_val_int8 = params["zero_point_int8"]
           zp_val_fp8 = params["zero_point_fp8"]
           print(f"{layer_name:<20} |Layer index:{layer_index:<7} |")
           print(f" └─scale type in INT8 Base: {scale_val_int8.dtype}, scale value:{scale_val_int8.item():<15.6f} ")
           print(f" └─scale type in F8 Base: {scale_val_fp8.dtype}, scale value:{scale_val_fp8.item():<15.6f} ")
           print(f" └─zero type in INT8 Base : {type(zp_val_int8)}, zero point: {zp_val_int8}")  
           print(f" └─zero type in F8 Base : {type(zp_val_fp8)}, zero point: {zp_val_fp8}") 
        
        

        print("=" * 60 + "\n")
        print("Compilation completion of scale and zero point of input for all layers")
        print("Compilation completely finished! Dictionary is ready for export.")
        print("#" * 60)
    except RuntimeError as e:
        print(f"\n [Critical Error] The complier crashed: {e}") # Catch the missing folder error gracefully if testing without images!

    return hardware_parameters_allLayers

# =============================================================
# SECTION 7: DEMO
# =============================================================

if __name__ == "__main__":
    # 1. Load a pre-trained model (e.g., ResNet18)
    model=models.resnet18(pretrained=True)
    model.eval() # critical: freeze BatchNorm and Dropout    
   
    # Run Layer 2 above calibration
    image_dir="./datasets/imagenette2-160/val"
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std  = [0.229, 0.224, 0.225] 
    
    try:    all_params_allLayers= calibrate_network(model=model,
            inputimage_dir=image_dir, dataset_mean=dataset_mean, 
            dataset_std=dataset_std, batch_size=32, 
            num_calibration_batches=8, Intbitwidth=8,
            strategy='percentile', pct= 99.9, precision="bf16",
            num_bins=2048, device='cpu', fp8_format='FP8_E4M3')
        
    except RuntimeError as e:
        print(f"\n [Crtical Error] The complier crashed: {e}") # Catch the missing folder error gracefully if testing without images!
        
    
