import os
from PIL import Image
import argparse

def convert_tiff_to_png(folder_name):
    for file in os.listdir(folder_name):
        if file.endswith(".tif"):
            new_file = os.path.join(folder_name, file.replace(".tif", ".png"))
            if not os.path.exists(new_file):
                img = Image.open(os.path.join(folder_name, file)).convert("RGB")
                img.save(new_file, "png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert TIFF images to PNG format.")
    parser.add_argument("folder_name", type=str, help="Path to the folder containing TIFF images.")
    args = parser.parse_args()

    convert_tiff_to_png(args.folder_name)