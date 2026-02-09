"""
PyTorch to TensorRT Converter - State Dict Compatible
Handles .pth files that contain only state_dict (OrderedDict)
"""

import torch
import torch.nn as nn
import torchvision
import tensorrt as trt
import os
import sys
from collections import OrderedDict
from ultralytics import YOLO

def check_pt_content(pt_path):
    """Check what's inside the .pt file"""
    print("Analyzing .pt file...")


    checkpoint = torch.load(pt_path, map_location='cpu') # , weights_only=True)
    
    if isinstance(checkpoint, OrderedDict):
        print("  ✓ File contains: state_dict (OrderedDict)")
        print("  ℹ You need to provide the model architecture")
        print("\n  Model layers found:")
        for i, (key, value) in enumerate(list(checkpoint.items())[:10]):
            print(f"    {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        if len(checkpoint) > 10:
            print(f"    ... and {len(checkpoint) - 10} more layers")
        return 'state_dict', checkpoint
    elif isinstance(checkpoint, dict):
        # Check for various common checkpoint formats
        if 'state_dict' in checkpoint:
            print("  ✓ File contains: checkpoint dict with 'state_dict' key")
            print("  ℹ You need to provide the model architecture")
            return 'checkpoint', checkpoint['state_dict']
        elif 'model' in checkpoint:
            print("  ✓ File contains: checkpoint dict with 'model' key")
            print(f"  ℹ Additional keys found: {list(checkpoint.keys())}")
            # Check if it's a complete model or just state_dict
            if isinstance(checkpoint['model'], nn.Module):
                print("  ℹ The 'model' key contains a complete nn.Module")
                return 'model', checkpoint['model']
            else:
                print("  ℹ The 'model' key contains a state_dict")
                return 'checkpoint', checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            print("  ✓ File contains: checkpoint dict with 'model_state_dict' key")
            return 'checkpoint', checkpoint['model_state_dict']
        else:
            # Might be a dict that is itself a state_dict
            print(f"  ⚠ Dict without standard keys. Keys: {list(checkpoint.keys())[:5]}")
            print("  ℹ Attempting to treat as state_dict...")
            return 'state_dict', checkpoint
    elif isinstance(checkpoint, nn.Module):
        print("  ✓ File contains: complete model")
        print("  ℹ No model architecture needed")
        return 'model', checkpoint
    else:
        print(f"  ⚠ Unknown format: {type(checkpoint)}")
        return 'unknown', checkpoint

def pt_to_trt(model_path, input_shape, model_architecture, device=None, fp16=True):
    """
    Convert PyTorch .pt model to TensorRT .trt engine
    
    Args:
        model_path: Output path for .trt file
        input_shape: Tuple of input shape (batch, channels, height, width)
        model_architecture: PyTorch model instance or class
        device: 'cuda' or 'cpu' (default: auto-detect)
        fp16: Enable FP16 precision for faster inference (requires CUDA)
    
    Returns:
        True if successful, False otherwise
    """
    trt_path = model_path + ".trt"
    pt_path = model_path + ".pt"
    
    try:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, falling back to CPU")
                device = 'cpu'
                fp16 = False
            else:
                # Test if CUDA is actually functional
                try:
                    torch.cuda.current_device()
                except Exception as e:
                    print(f"WARNING: CUDA is available but not functional ({e})")
                    print("         Falling back to CPU for ONNX export")
                    device = 'cpu'
                    fp16 = False
        
        print(f"\n{'='*60}")
        print(f"Converting {pt_path} to {trt_path}")
        print(f"Device: {device}")
        print(f"Input shape: {input_shape}")
        print(f"FP16 mode: {fp16}")
        print(f"{'='*60}\n")
        
        # Step 1: Load PyTorch model
        print("Step 1/3: Loading PyTorch model...")
        
        # Check what's in the file
        content_type, content = check_pt_content(pt_path)
        
        if content_type == 'model':
            # Complete model loaded
            model = content
        elif content_type in ['state_dict', 'checkpoint']:
            # Need to load into architecture
            if model_architecture is None:
                print("\n✗ Error: state_dict found but no model architecture provided!")
                print("\nYou need to define your model architecture and pass it as model_architecture parameter.")
                print("See the example at the bottom of this script.")
                return False
            
            # Initialize model
            if isinstance(model_architecture, type):
                # If it's a class, instantiate it
                model = model_architecture()
            else:
                # Already an instance
                model = model_architecture
            
            # Load state dict
            print("  Loading state_dict into model architecture...")
            model.load_state_dict(content)
        else:
            print(f"\n✗ Error: Unknown .pt file format: {type(content)}")
            return False
        
        model.eval()
        
        # Convert model to FP32 to avoid dtype mismatches during ONNX export
        model = model.float()
        model.to(device)
        print("  ✓ Model loaded successfully (converted to FP32 for export)")
        
        # Step 2: Export to ONNX
        onnx_path = trt_path.replace('.trt', '.onnx')
        print(f"\nStep 2/3: Exporting to ONNX ({onnx_path})...")
        
        dummy_input = torch.randn(input_shape).to(device)
        
        # Use legacy TorchScript-based exporter for compatibility
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['l_x_'],        # Must match C++ runtimes.cpp
                output_names=['sigmoid_1'],  # Must match C++ runtimes.cpp
                dynamic_axes={
                    'l_x_': {0: 'batch'},
                    'sigmoid_1': {0: 'batch'}
                },
                verbose=False# ,
                # dynamo=False  # Use legacy exporter
            )
        print("  ✓ ONNX export successful")
        
        # Step 3: Build TensorRT engine
        print("\nStep 3/3: Building TensorRT engine...")
        
        # Check if we can build TensorRT engine (requires CUDA)
        if device == 'cpu':
            print("  ⚠ WARNING: Cannot build TensorRT engine on CPU")
            print("  ⚠ TensorRT requires GPU support for engine building")
            print("  ✓ ONNX model has been exported successfully")
            print(f"\n{'='*60}")
            print("✓ PARTIAL SUCCESS - ONNX Export Complete")
            print(f"{'='*60}\n")
            print(f"ONNX model saved to: {onnx_path}")
            print(f"File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
            print("\nNote: To build a TensorRT engine, you need a system with CUDA GPU support.")
            print("      The ONNX model can be used for inference on CPU or converted on a GPU system.")
            return True
        
        print("  This may take a few minutes...")
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
        except Exception as e:
            print(f"  ✗ Failed to create TensorRT builder: {e}")
            print("  ⚠ This typically means CUDA is not available on this system")
            print("  ✓ ONNX model has been exported successfully")
            print(f"\n{'='*60}")
            print("✓ PARTIAL SUCCESS - ONNX Export Complete")
            print(f"{'='*60}\n")
            print(f"ONNX model saved to: {onnx_path}")
            return True
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        print("  Parsing ONNX model...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("  ✗ Failed to parse ONNX file:")
                for error in range(parser.num_errors):
                    print(f"    Error {error}: {parser.get_error(error)}")
                return False
        
        print("  ✓ ONNX parsed successfully")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Add optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        # Set min, optimal, and max batch sizes (using input shape from parameter)
        min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        max_shape = (8, input_shape[1], input_shape[2], input_shape[3])
        profile.set_shape("l_x_", min_shape, opt_shape, max_shape)  # Must match input name
        config.add_optimization_profile(profile)
        print("  ✓ Optimization profile added")
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 mode enabled")
        elif fp16:
            print("  ⚠ FP16 requested but not supported on this platform")
        
        # Build engine
        print("  Building engine (this is the slow part)...")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            print("  ✗ Failed to build TensorRT engine")
            return False
        
        print("  ✓ Engine built successfully")
        
        # Save engine (build_serialized_network returns bytes directly)
        print(f"  Saving engine to {trt_path}...")
        with open(trt_path, 'wb') as f:
            f.write(engine)
        
        print("  ✓ Engine saved successfully")
        
        # Clean up ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"  ✓ Cleaned up intermediate ONNX file")
        
        print(f"\n{'='*60}")
        print("✓ CONVERSION SUCCESSFUL!")
        print(f"{'='*60}\n")
        print(f"TensorRT engine saved to: {trt_path}")
        print(f"File size: {os.path.getsize(trt_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {type(e).__name__}")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# DEFINE YOUR MODEL ARCHITECTURE HERE
# ============================================================================

class ClassifierEfficient(nn.Module):
    """
    EfficientNet-B0 based classifier for region classification.
    40 output classes for different geographic regions.
    """
    def __init__(self, num_classes=40):
        super(ClassifierEfficient, self).__init__()

        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = torchvision.models.efficientnet_b0(weights=weights)
        for param in self.efficientnet.features[:3].parameters():
            param.requires_grad = False
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========== CONFIGURATION - MODIFY THESE VALUES ==========
    MODEL_PATH = "rc_model_weights"
    INPUT_SHAPE = (1, 3, 224, 224)  # (batch_size, channels, height, width)
    
    # Create model architecture instance
    # If your .pth contains only state_dict, you MUST provide the architecture
    model_architecture = ClassifierEfficient()
    
    # If your .pth contains the complete model, set this to None:
    # model_architecture = None
    ld_folder = "models/V1/trained-ld/"

    list_folder = os.listdir(ld_folder)
    print(list_folder)

    for folder in list_folder:
        if not os.path.isdir(os.path.join(ld_folder, folder)):
            continue
        path = os.path.join(ld_folder, folder, f"{folder}_nadir")
        print(f"Converting model at: {path}")
        pt_to_trt(
            model_path=path,
            input_shape=INPUT_SHAPE,
            model_architecture=model_architecture,  # Your model architecture
            device=None,  # Auto-detect
            fp16=True     # Enable FP16
        )
