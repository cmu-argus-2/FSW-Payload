import torch
import sys
import torch


if __name__ == "__main__":
    try:
        model = torch.jit.load(sys.argv[1])
        print("The model is in TorchScript format.")
    except RuntimeError as e:
        print("The model is not in TorchScript format.")
