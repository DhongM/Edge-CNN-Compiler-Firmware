import torch
import torchvision.models as models
import torch.nn as nn
from typing import Dict, Optional, List, Any
import torch.fx
from collections import deque

def extract_nodes(arg:Any)->List[torch.fx.Node]:
    """
    Recursively extracts all PyTorch FX Nodes from nested tuples, lists, or dicts.
    This safely handles complex inputs like torch.cat([a, b], dim=1).
    """
    nodes=[]
    if isinstance(arg, torch.fx.Node):
        nodes.append(arg)
    elif isinstance(arg, (tuple,list)):
        for item in arg:
            nodes.extend(extract_nodes(item))
    elif isinstance(arg,dict):
        for val in arg.values():
            nodes.extend(extract_nodes(val))
    return nodes
    
def find_source_in_calib(node: torch.fx.Node,
         calib_dict:Dict[str,dict], max_depth: int=5)->Optional[str]:
    """
    Performs a Breadth-First Search (BFS) backwards through the FX graph.
    Explores ALL incoming wires (args/kwargs) to find the nearest calibrated node.
    
    Handles ALL node types:
      placeholder  : model input → 'layer1_input'
      call_module  : nn.Module (ReLU, MaxPool, Identity...)
      call_function: torch.add, torch.cat, torch.flatten...
      call_method  : tensor.view(), tensor.mean()...
      
    Handles ALL graph topologies:
      Sequential CNN:    single path A → straightforward
      Residual (add):    path A + path B → searches both
      DenseNet (concat): path A + B + C → searches all
      Args:
        node : input node to Conv2d/Linear (node.args[0])
        calib_dict : {layer_name: {scale, zero_point}}
        max_depth  : max BFS depth (prevents infinite loops)
    Returns:
        source_name : key in calib_dict, or None if not found
    """
    # BFS queue: (node, depth)
    # Queue stores tuples of: (current_node, current_depth)
    queue=deque([(node, 0)])
    # Keep track of visited nodes so don't get stuck in graph loops!
    # Use {node} to safely initialize a set with one object
    visited={node}    # prevent revisiting same node
    
    while queue:
         current, depth=queue.popleft()
         # ── 1. Check if we found a match (Base Cases) ────────────
         if current.op=='placeholder':
             # This is Layer 1 — input is raw image
            # 'layer1_input' is the special calib_dict key
             return 'layer1_input' # Layer 1 — input is raw image
         elif current.op=='call_module' and current.target in calib_dict:
             return current.target  # Found calibrated module ✅
         # ── 2. Enforce Depth Limit ──────────────────────────
         elif depth>max_depth:
             continue 
         # ── 3. Find ALL incoming wires to this node ──────────────────
         incoming_nodes=extract_nodes(current.args)+extract_nodes (current.kwargs)  
         for incoming_node in incoming_nodes:
             if incoming_node not in visited: 
                visited.add(incoming_node)
                queue.append((incoming_node, depth+1))
    return None # Exceeded max_depth or hit a dead end
             

def dynamic_topology_mapper(model:torch.nn.Module, hardware_payload:dict,
                    calib_dict):         
    """
    Dynamically traces ANY PyTorch model's computational graph to find 
    exactly which activation layer feeds into which Conv2d/Linear layer.
    """
    print("\n[Topology Mapper] Tracing Computational Graph with PyTorch FX...")
    # 1. Trace the model to get the wire graph
    tracer=torch.fx.symbolic_trace(model)
    modules=dict(tracer.named_modules())
    mapping = {}
    unmapped_list=[]
    #QUANT_BLACKLIST=['fc','classifier','head'] # Layers to keep in FP8
    # 2. Walk through the nodes in the graph
    for node in tracer.graph.nodes: 
        if node.op!='call_module':
            continue
        target_module=modules[node.target]
        if target_module is None:
            continue 
        #  Only attempt to map Convolutions and Linears!
        if not isinstance(target_module,(nn.Conv2d, nn.Linear)):
            continue 
        conv_name=node.target
        # node.args[0] is the primary input activation
        # But node may have more args in some architectures
        # search backwards from ALL input args
        source_name=None
        for arg in node.args:
            found=find_source_in_calib(node=arg,
                        calib_dict=calib_dict, max_depth=5)
            if found is not None:
                source_name=found
                break 

        if  conv_name in hardware_payload and source_name is not None\
            and source_name in calib_dict:
            
            mapping[conv_name] = source_name 
            is_fp8 = isinstance(target_module, nn.Linear)
            path   = 'FP8 ' if is_fp8 else 'INT8' 
            # --- THE BRILLIANT DYNAMIC ROUTER ---
            # For all layers, dynamically pick INT8 or FP8 based on the destination!
            if is_fp8:
                hardware_payload[conv_name]["input_scale"] = calib_dict[source_name]["scale_fp8"]
                hardware_payload[conv_name]["input_zero_point"] = calib_dict[source_name]["zero_point_fp8"]
            else:
                hardware_payload[conv_name]["input_scale"] = calib_dict[source_name]["scale_int8"]
                hardware_payload[conv_name]["input_zero_point"] = calib_dict[source_name]["zero_point_int8"]    

             
            print(f"  ├─ Auto-Linked: {path}: {source_name:<15} -> {conv_name:<20}")
            
            w_tensor = hardware_payload[conv_name]['quantizedweights_channel']
            w_scale  = hardware_payload[conv_name]['weight_scale']
            w_zp     = hardware_payload[conv_name]['zero_point']
            i_scale  = hardware_payload[conv_name]['input_scale']
            i_zp     = hardware_payload[conv_name]['input_zero_point']

            # Safely handle zero_point types (Tensor vs int)
            #w_zp_type = w_zp.dtype if torch.is_tensor(w_zp) else type(w_zp)     
            print(f"  ├─ Hardware Weights : {w_tensor.dtype} (Shape: {tuple(w_tensor.shape)})")
            print(f"  ├─ Weight Scales    : {w_scale.dtype}, (minVale:{w_scale.min().item():.6f}), (maxVale:{w_scale.max().item():.6f})")
            # SAFELY handle the zero point (Only prints if it exists, avoids KeyError on FP8)
            w_zp = hardware_payload[conv_name].get('zero_point')
            if  w_zp is not None:
                if torch.is_tensor(w_zp):
                    # It's an INT8 layer (Tensor)
                    print(f"  ├─ Weight ZP        : {w_zp.dtype}, (minVale: {w_zp.min().item()}), (maxVale: {w_zp.max().item()})")
                else:
                    # It's an FP8 layer (Python Integer)
                    print(f"  ├─ Weight ZP        : {type(w_zp)}, (Value: {w_zp})")
            else:
                print(f"  ├─ Weight ZP : N/A ")

            print(f"  ├─ Input Scales : {i_scale.dtype}, (Vale:{i_scale.item():.6f})")
            print(f"  └─ Input ZP     : {type(i_zp)} (Value: {i_zp})\n")

        else:
            unmapped_list.append((source_name, conv_name))
            reason = []
            if conv_name not in hardware_payload:
                reason.append(f"{conv_name} not in hardware_payload")
            if source_name is None:
                reason.append("no calibrated source found in graph")
            elif source_name not in calib_dict: 
                reason.append(f"{source_name} not in calib_dict")
            print(f"  ⚠️  Unmapped: {source_name} -> {conv_name}")
            print(f" Reason:{','.join(reason)}")

    print(f"\n  ✅ Mapped   : {len(mapping)} layers")
    print(f"  ⚠️  Unmapped : {len(unmapped_list)}")
    return hardware_payload, mapping


