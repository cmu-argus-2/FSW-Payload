import onnx
import onnx_graphsurgeon as gs
import numpy as np

# this is untested code to fix the .onnx file from "In node 7 (importConv): 
# INVALID_NODE: Assertion failed: (nchan == -1 || kernelWeights.shape.d[1] * ngroup == nchan) && "Kernel weight dimension failed to broadcast to input."""
# Primary printouts of the structure of the model -> printing out reduce mean axes arguments

# Load model
model = onnx.load("rc_model_weights.onnx")
graph = gs.import_onnx(model)

patched = False

for node in graph.nodes:
    if node.op == "ReduceMean":
        print(node)
        print(node.inputs[1].values)
        axes = node.attrs.get("axes", None)

        # Target the broken SE ReduceMean
        if axes == [2]:
            print(f"Fixing ReduceMean node: {node.name}")

            node.attrs["axes"] = [2, 3]
            node.attrs["keepdims"] = 1
            patched = True

# Clean up graph
graph.cleanup().toposort()

if not patched:
    raise RuntimeError("No ReduceMean node with axes=[2] found")

# Save fixed model
onnx.save(gs.export_onnx(graph), "model_fixed.onnx")
print("Saved model_fixed.onnx")
