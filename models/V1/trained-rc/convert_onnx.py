import os
import torch
from run_basic_rc import ClassifierEfficient

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierEfficient().to(device)
    model_weights_path = os.path.join("effnet_0.997acc.pth")
    # Load Custom model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Input tensor shape (images)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)


    torch.onnx.export(
        model,                  # model to export
        (input_tensor,),        # inputs of the model,
        "effnet.onnx",        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True             
    )