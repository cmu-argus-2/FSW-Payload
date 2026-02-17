import os

# import cv2
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import tensorrt as trt
import numpy as np

"""
'10S': 'California',
'10T': 'Washington / Oregon',
'11R': 'Baja California, Mexico',
'12R': 'Sonora, Mexico',
'16T': 'Minnesota / Wisconsin / Iowa / Illinois',
'17R': 'Florida',
'17T': 'Toronto, Canada / Michigan / OH / PA',
'18S': 'New Jersey / Washington DC',
'32S': 'Tunisia (North Africa near Tyrrhenian Sea)',
'32T': 'Switzerland / Italy / Tyrrhenian Sea',
'33S': 'Sicilia, Italy',
'33T': 'Italy / Adriatic Sea',
'52S': 'Korea / Kumamoto, Japan',
'53S': 'Hiroshima to Nagoya, Japan',
'54S': 'Tokyo to Hachinohe, Japan',
'54T': 'Sapporo, Japan'
"""

class ClassifierEfficient(nn.Module):

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

if __name__ == "__main__":

    # Load Custom model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierEfficient().to(device)
    model_weights_path = os.path.join(os.path.dirname(__file__), "effnet_0.997acc.pth")
    # model_weights_path = os.path.join(os.path.dirname(__file__), "/home/argus/Documents/models/region_classifier/model10.pth")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    
    trt_model_weights_path = os.path.join(os.path.dirname(__file__), "rc_model_weights.trt")
    
    with open(trt_model_weights_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Mapping of 16 regions
    idx_mapping = [
        "10S",
        "10T",
        "11R",
        "12R",
        "16T",
        "17R",
        "17T",
        "18S",
        "32S",
        "32T",
        "33S",
        "33T",
        "52S",
        "53S",
        "54S",
        "54T",
    ]

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    folder_name = os.path.join(os.path.dirname(__file__), "/home/argus/Documents/test_photos/17R")
    files = os.listdir(folder_name)
    results = {}

    for file in files:
        if not file.endswith((".png", ".jpg", ".jpeg")):
            continue
        img = Image.open(os.path.join(folder_name, file)).convert("RGB")
        #img = Image.open("tests/sample_data/inference/l9_32S_00001.png").convert("RGB")
        # img = Image.open("tests/sample_data/inference/l9_10T_00021.png").convert("RGB")
        img = transformations(img).unsqueeze(0).to(device)

        print(img.shape)

        print(f"Input tensor shape: {img.shape}")
        #print("First 10 pixels of the input tensor:")
        #print(img[0, :, :10, :10])
        outputs = model(img)
        print(f"Output tensor: {[f'{v:.3f}' for v in outputs.detach().cpu().numpy().flatten()]}")
        predicted = torch.where(outputs > 0.5)[1]

        # results[file] = [idx_mapping[p] for p in predicted]
        #print(f"{file}")
        print(f"output: {[f'{v:.3f}' for v in outputs.detach().cpu().numpy().flatten()]}")
        print(f"predicted indices: {predicted.cpu().numpy().flatten().tolist()}")
        # print(f"prediction: {results[file]}")
        print("----------")
        
        # run the rc net tensorrt
        # Prepare input for TensorRT
        image_array = img.cpu().numpy()
        input_data = np.ascontiguousarray(image_array.flatten(), dtype=np.float32)
        output_data = np.empty((1,40), dtype=np.float32)
        
        bindings = [int(input_data.ctypes.data), int(output_data.ctypes.data)]
        
        context.execute_v2(bindings)
        print(f"TensorRT output: {[f'{v:.3f}' for v in output_data.flatten()]}")
        predicted_trt = np.where(output_data > 0.5)[1]
        print(f"TensorRT predicted indices: {predicted_trt.flatten().tolist()}")
        print("==========")
        
        # exit()

