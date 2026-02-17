import os
import torch
from run_sample_rc import ClassifierEfficient

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierEfficient(num_classes=16).to(device) # Make sure to change this back later and same in rc_sample_rc.py
    model_weights_path = os.path.join("/home/argus/Documents/batch_opt/FSW-Payload-2/models/V1/trained-rc/effnet_0.997acc.pth")
    # Load Custom model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Input tensor shape (images)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # NOTE: Changes were made 2/16/26 to the onnx file. Newer versions of onnx (with opset_version >= 18) have conversion issues with tensorRT (8.6.X) and cuda-toolkit (12.2)
    # Dynamo should be false, opset should be set to 17
    torch.onnx.export(
        model,                  # model to export
        (input_tensor,),        # inputs of the model,
        "effnet_0997acc.onnx",        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )
