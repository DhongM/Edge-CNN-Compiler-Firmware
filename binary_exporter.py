"""
export_to_binary.py
=======================================================
Master AI Hardware Binary Exporter.
Combines Sectioned DMA Architecture with Bare-Metal PyTorch Serialization.

model.bin layout (Little-Endian):
  FILE HEADER (32 bytes)
  SECTION 1: WEIGHTS →(W-SRAM) Weight SRAM
  SECTION 2: BIAS   → (O-SRAM) Bias SRAM
  SECTION 3: CALIB  →  (Vector Registers) Register File
  manifest.json → stays on host, NOT sent over PCIe
  Human-readable index for debugging and validation
"""
import os
import struct
import torch
import numpy as np
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import torch.nn as nn

# =============================================================
# SECTION 1: CONSTANTS (Hardware Magic Numbers)
# =============================================================
MAGIC_MODEL       = 0x4D4F444C   # 'MODL'
MAGIC_WEIGHTS     = 0x57454947   # 'WEIG'
MAGIC_BIAS        = 0x42494153   # 'BIAS'
MAGIC_CALIBRATION = 0x43414C49   # 'CALI'
FILE_VERSION      = 1
ALIGN_BYTES       = 8

TAG_INT8_CONV     = 0x01
TAG_FP8_LINEAR    = 0x02
TAG_INT32_BIAS    = 0x11
TAG_FP16_BIAS     = 0x12
TAG_INT8_INPUT    = 0x21
TAG_FP8_INPUT     = 0x22

FP8_E4M3          = 0x01
FP8_E5M2          = 0x02



# =============================================================
# SECTION 2: HARDWARE MEMORY HELPERS
# =============================================================
def pad_to_align(data: bytes, align: int = ALIGN_BYTES) -> bytes:
    """Pads a byte array with zeros so it cleanly aligns with hardware bus widths."""
    remainder = len(data) % align
    if remainder == 0:
        return data
    return data + b'\x00' * (align - remainder)

def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Safely extracts raw memory bytes, dodging PyTorch's bfloat16 limitations.
    Convert any tensor to raw bytes. Handles all dtypes"""
    t = tensor.contiguous().cpu()
    # If the hardware expects bfloat16 or FP8, we pull raw bits without modifying them
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

# =============================================================
# SECTION 3: DMA SECTION BUILDERS
# =============================================================
def build_weights_section(hardware_payload: Dict[str, dict]) -> Tuple[bytes, list]:
    """
    Build complete weights section as bytes.
    Returns (weights section_bytes, layer_info_list)
    """
    buf = bytearray()
    layer_info = []
    
    # <I4x = Little-Endian Unsigned Int + 4 bytes padding (8 bytes total)
    buf += struct.pack('<I4x', MAGIC_WEIGHTS) 

    for name, payload in hardware_payload.items():
        if "quantizedweights_channel" not in payload:
            continue

        name_bytes = name.encode('utf-8')
        w_tensor = payload["quantizedweights_channel"]
        w_scale  = payload["weight_scale"]
        
        # Determine Layer Type
        is_fp8 = w_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        tag = TAG_FP8_LINEAR if is_fp8 else TAG_INT8_CONV
        
        # Safely handle 4D (Conv) vs 2D (Linear) shapes
        shape = w_tensor.shape
        out_ch, in_ch = shape[0], shape[1]
        kH = shape[2] if w_tensor.dim() == 4 else 1
        kW = shape[3] if w_tensor.dim() == 4 else 1

        # Descriptor: 32 bytes total. (Tag, NameLen, Out, In, kH, kW, reserved padding)
        # Little Endian '<' applied!
        desc = struct.pack('<BBHHBBBB22x', tag, len(name_bytes), out_ch, in_ch, kH, kW, 0, 0)
        
        buf += desc
        buf += pad_to_align(name_bytes)
        buf += pad_to_align(tensor_to_bytes(w_tensor))
        buf += pad_to_align(tensor_to_bytes(w_scale))
        
        # Only pack ZP array if INT8. (FP8 assumes 0, handled in calib section)
        #w_zp = payload.get("zero_point")
        if not is_fp8:
            w_zp = payload.get("zero_point")
            if torch.is_tensor(w_zp):
                buf += pad_to_align(tensor_to_bytes(w_zp))
        else:
            # Force a 32-bit integer '0' into the binary for FP8
            buf += pad_to_align(struct.pack('<i', 0)) 

        layer_info.append({'name': name, 'type': 'FP8' if is_fp8 else 'INT8', 'shape': list(shape)})

    return bytes(buf), layer_info

def build_bias_section(hardware_payload: Dict[str, dict]) -> Tuple[bytes, list]:
    """Build complete bias section as bytes."""
    buf = bytearray()
    layer_info = []
    buf += struct.pack('<I4x', MAGIC_BIAS)

    for name, payload in hardware_payload.items():
        bias_tensor = payload.get("bias")
        if bias_tensor is None:
            continue

        name_bytes = name.encode('utf-8')
        record_info = {'name': name, 'offset': len(buf)}

        out_ch = bias_tensor.shape[0]
        
        is_int32 = bias_tensor.dtype == torch.int32
        tag = TAG_INT32_BIAS if is_int32 else TAG_FP16_BIAS
        desc = struct.pack('<BBHH26x', tag, len(name_bytes), out_ch, 0)
        buf += desc
        buf += pad_to_align(name_bytes)
        buf += pad_to_align(tensor_to_bytes(bias_tensor))
        if is_int32:
            bias_scale = payload["bias_scale"]
            buf += pad_to_align(tensor_to_bytes(bias_scale))
            
            record_info.update({
                'type': 'INT32', 
                'out_ch': out_ch,
                'bias range': [bias_tensor.min().item(), bias_tensor.max().item()],
                'bias_Scale range': [bias_scale.min().item(), bias_scale.max().item()]
            })
            layer_info.append(record_info)
        else:
            # FIXED: Added '<' for Little Endian
            buf += pad_to_align(struct.pack('<i', 0))
            
            record_info.update({
                'type': str(bias_tensor.dtype), 
                'out_ch': out_ch,
                'bias range': [bias_tensor.float().min().item(), bias_tensor.float().max().item()]
            })
            layer_info.append(record_info)

        #layer_info.append({'name': name, 'type': 'INT32' if is_int32 else str(bias_tensor.dtype)})

    return bytes(buf), layer_info

def build_calibration_section(hardware_payload: Dict[str, dict]) -> Tuple[bytes, list]:
    buf = bytearray()
    layer_info = []
    buf += struct.pack('<I4x', MAGIC_CALIBRATION)

    for idx, (name, payload) in enumerate(hardware_payload.items()):
        if 'input_scale' not in payload:
            continue

        name_bytes  = name.encode('utf-8')
        i_scale     = payload['input_scale']
        i_zp        = payload['input_zero_point']
        
        w_tensor    = payload.get("quantizedweights_channel")
        is_fp8      = False if w_tensor is None else (w_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2])
        tag         = TAG_FP8_INPUT if is_fp8 else TAG_INT8_INPUT

        desc = struct.pack('<BBHH26x', tag, len(name_bytes), idx + 1, 0)
        buf += desc
        buf += pad_to_align(name_bytes)
        buf += pad_to_align(tensor_to_bytes(i_scale))

        # Pack the Input ZP as a signed 32-bit int ('<i') and align it, zp_fp8 = 0
        buf += pad_to_align(struct.pack('<i', int(i_zp)))

        layer_info.append({'name': name, 
                           'scale_dtype': str(i_scale.dtype),
                           'input_scale':  i_scale.item(),  # Extracts standard float
                           'input_zero_point':  int(i_zp)        # Extracts standard int
                          })

    return bytes(buf), layer_info
# =============================================================
# SECTION 4: MAIN EXPORT FUNCTION
# =============================================================
@dataclass
class ExportSummary:
    model_bin_path:  str
    manifest_path:   str
    weights_bytes:   int
    bias_bytes:      int
    calib_bytes:     int
    total_bytes:     int
    sha256:          str

    def print(self):
        print(f"\n {'─'*60}")
        print(f" 💾 model.bin    : {self.model_bin_path}")
        print(f" 📄 manifest.json: {self.manifest_path}")
        print(f" {'─'*60}")
        print(f"  Weights Buffer (W-SRAM)  : {self.weights_bytes:>10,} B")
        print(f"  Bias Buffer (Bias-SRAM)  : {self.bias_bytes:>10,} B")
        print(f"  Vector Registers (Calibration) : {self.calib_bytes:>10,} B")
        print(f"  TOTAL PAYLOAD SIZE       : {self.total_bytes:>10,} B")
        print(f"  SHA-256 Checksum         : {self.sha256[:16]}...")
        print(f" {'─'*60}")
        print(f"  Sent ONCE over PCIe at model load time ✅")
        print(f"  manifest.json stays on HOST for debugging ✅")

def export_model_bin(hardware_payload: Dict[str, dict], output_dir: str = './hardware_bins', 
                     model_name: str = 'resnet18_firmware') -> ExportSummary:
    """Export all quantized data to ONE combined model.bin file.
    Exports the hardware payload to a sectioned, DMA-ready binary format
    Also writes manifest.json for debugging (not sent over PCIe).
    model.bin layout:
      FILE HEADER (32 bytes)
        magic, version, n_layers
        weights_offset, bias_offset, calib_offset
        total_size
      SECTION 1: WEIGHTS
      SECTION 2: BIAS
      SECTION 3: CALIBRATION
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_path      = os.path.join(output_dir, f'{model_name}.bin')
    manifest_path = os.path.join(output_dir, f'{model_name}_manifest.json')

    print("\n" + "=" * 60)
    print(" 🚀 BOOTING BINARY EXPORTER — Generating Silicon Firmware")
    print("=" * 60)

    # 1. Build Sections
    w_bytes, w_info = build_weights_section(hardware_payload)
    b_bytes, b_info = build_bias_section(hardware_payload)
    c_bytes, c_info = build_calibration_section(hardware_payload)

    print(f"  Weights section     : {len(w_bytes):>8,} bytes"
          f"  ({len(w_info)} layers)")
    print(f"  Bias section        : {len(b_bytes):>8,} bytes"
          f"  ({len(b_info)} layers)")
    print(f"  Calibration section : {len(c_bytes):>8,} bytes"
          f"  ({len(c_info)} layers)")

    # 2. Compute section offsets for FILE HEADER ────────────────
    HEADER_SIZE    = 32
    weights_offset = HEADER_SIZE
    bias_offset    = weights_offset + len(w_bytes)
    calib_offset   = bias_offset    + len(b_bytes)
    total_size     = calib_offset   + len(c_bytes)

    n_layers       = len(w_info)

    # 3. Write Combined Binary
    with open(bin_path, 'wb') as f:
        # FILE HEADER (32 bytes) - Little Endian '<'
        # magic(4) version(2) n_layers(2) reserved(8)
        # weights_offset(4) bias_offset(4) calib_offset(4)
        # total_size(4) reserved(4)
        header = struct.pack(
            '<IHHQ IIII',
            MAGIC_MODEL,      # 4B magic
            FILE_VERSION,     # 2B version
            n_layers,         # 2B num layers
            0,                # 8B reserved
            weights_offset,   # 4B weights offset
            bias_offset,      # 4B bias offset
            calib_offset,     # 4B calib offset
            total_size,       # 4B total size
        )
        f.write(header.ljust(32, b'\x00')) # Ensure exactly 32 bytes

        # Write the core sections,Three sections back to back
        f.write(w_bytes)
        f.write(b_bytes)
        f.write(c_bytes)
    print(f"\n  ✅ model.bin written: {total_size:,} bytes")  

    # 4. Generate JSON Manifest.json (host only, not sent over PCIe)

    sha = sha256_file(bin_path)
    manifest = {
        'version': FILE_VERSION,
        'model_name': model_name,
        'sha256': sha,
        'total_bytes': total_size,
        'section_offsets': {'weights': weights_offset, 
                            'bias': bias_offset, 
                            'calibration': calib_offset},
        'section_sizes': {
            'weights':      len(w_bytes),
            'bias':         len(b_bytes),
            'calibration':  len(c_bytes),
        },
        'weights_layers':     w_info,
        'bias_layers':        b_info,
        'calibration_layers': c_info,

    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✅ manifest.json written (host only, debug use)")
    print(f"  SHA256: {sha[:32]}...")

    # 5. Output Summary
    summary = ExportSummary(
        model_bin_path=bin_path, manifest_path=manifest_path,
        weights_bytes=len(w_bytes), bias_bytes=len(b_bytes),
        calib_bytes=len(c_bytes), total_bytes=total_size, sha256=sha
    )
    summary.print()
    return summary

