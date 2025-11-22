import argparse
import os
from ultralytics import YOLO

def convert_model(path, **config):
    model = YOLO(path, task='obb') 
    model.export(**config)  


if __name__ == "__main__":

    # https://docs.ultralytics.com/modes/export/#arguments
    parser = argparse.ArgumentParser(description="Convert YOLO models.")
    parser.add_argument("--format", type=str, default="torchscript", help="Format to convert the models to")
    #parser.add_argument("--imgsz", type=int, default=1216, help="Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.")
    parser.add_argument("--half", type=bool, default=False, help="Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.")
    parser.add_argument("--int8", type=bool, default=False, help="Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.")
    parser.add_argument("--batch", type=int, default=1, help="Specifies export model batch inference size or the max number of images the exported model will process concurrently in predict mode.")
    parser.add_argument("--optimize", type=bool, default=True, help="Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.")
    
    
    config = vars(parser.parse_args())

    ld_folder = "ld"

    list_folder = os.listdir(ld_folder)
    print(list_folder)

    for folder in list_folder:
        path = os.path.join(ld_folder, folder, f"{folder}_nadir.pt")
        print(f"Converting model at: {path}")
        convert_model(path, **config)  
        break
