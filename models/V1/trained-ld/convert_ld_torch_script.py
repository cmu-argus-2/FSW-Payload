import argparse
import os
from ultralytics import YOLO

def convert_model(path, **config):
    model = YOLO(path) # , task='obb') 
    model.export(**config)  


if __name__ == "__main__":

    # https://docs.ultralytics.com/modes/export/#arguments
    parser = argparse.ArgumentParser(description="Convert YOLO models.")
    parser.add_argument("--format", type=str, default="engine", help="Format to convert the models to")
    #parser.add_argument("--imgsz", type=int, default=1216, help="Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.")
    parser.add_argument("--half", type=bool, default=False, help="Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.")
    parser.add_argument("--int8", type=bool, default=False, help="Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.")
    parser.add_argument("--batch", type=int, default=1, help="Specifies export model batch inference size or the max number of images the exported model will process concurrently in predict mode.")
    parser.add_argument("--optimize", type=bool, default=False, help="Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.")
    parser.add_argument("--nms", type=bool, default=True, help="Adds Non-Maximum Suppression (NMS) to the exported model when supported, improving detection post-processing efficiency.")
    parser.add_argument("--device", type=str, default=0, help="Specifies the device for exporting: GPU (device=0), CPU (device=cpu), MPS for Apple silicon (device=mps) or DLA for NVIDIA Jetson (device=dla:0 or device=dla:1). TensorRT exports automatically use GPU.")
    parser.add_argument("--imgsz", type=int, default=640, help="Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.")
    parser.add_argument("--dynamic", type=bool, default=True, help="Adds Non-Maximum Suppression (NMS) to the exported model when supported, improving detection post-processing efficiency.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enables verbose logging during export, providing detailed information about the export process and any potential issues.")
    # parser.add_argument("--workspace", type=int, default=1, help="Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance. Use None for auto-allocation by TensorRT up to device maximum.")
    config = vars(parser.parse_args())

    ld_folder = os.path.dirname(os.path.abspath(__file__))
    print(f"Looking for models in folder: {ld_folder}")
    list_folder = os.listdir(ld_folder)
    print(list_folder)

    for folder in list_folder:
        if not os.path.isdir(os.path.join(ld_folder, folder)) or not folder.startswith("10T"):
            continue
        path = os.path.join(ld_folder, folder, f"{folder}_weights.pt")
        # config["data"] = os.path.join(ld_folder, folder, "dataset.yaml")
        print(f"Converting model at: {path}")
        convert_model(path, **config)
        break
        
"""
format	str	'torchscript'	Target format for the exported model, such as 'onnx', 'torchscript', 'engine' (TensorRT), or others. Each format enables compatibility with different deployment environments.
imgsz	int or tuple	640	Desired image size for the model input. Can be an integer for square images (e.g., 640 for 640×640) or a tuple (height, width) for specific dimensions.
keras	bool	False	Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.
optimize	bool	False	Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving inference performance. Not compatible with NCNN format or CUDA devices.
half	bool	False	Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware. Not compatible with INT8 quantization or CPU-only exports. Only available for certain formats, e.g. ONNX (see below).
int8	bool	False	Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices. When used with TensorRT, performs post-training quantization (PTQ).
dynamic	bool	False	Allows dynamic input sizes for TorchScript, ONNX, OpenVINO, TensorRT, and CoreML exports, enhancing flexibility in handling varying image dimensions. Automatically set to True when using TensorRT with INT8.
simplify	bool	True	Simplifies the model graph for ONNX exports with onnxslim, potentially improving performance and compatibility with inference engines.
opset	int	None	Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.
workspace	float or None	None	Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance. Use None for auto-allocation by TensorRT up to device maximum.
nms	bool	False	Adds Non-Maximum Suppression (NMS) to the exported model when supported (see Export Formats), improving detection post-processing efficiency. Not available for end2end models.
batch	int	1	Specifies export model batch inference size or the maximum number of images the exported model will process concurrently in predict mode. For Edge TPU exports, this is automatically set to 1.
device	str	None	Specifies the device for exporting: GPU (device=0), CPU (device=cpu), MPS for Apple silicon (device=mps) or DLA for NVIDIA Jetson (device=dla:0 or device=dla:1). TensorRT exports automatically use GPU.
data	str	'coco8.yaml'	Path to the dataset configuration file, essential for INT8 quantization calibration. If not specified with INT8 enabled, coco8.yaml will be used as a fallback for calibration.
fraction	float	1.0	Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used.
end2end	bool	None	Overrides the end-to-end mode in YOLO models that support NMS-free inference (YOLO26, YOLOv10). Setting it to False lets you export these models to be compatible with the traditional NMS-based postprocessing pipeline.
""" 