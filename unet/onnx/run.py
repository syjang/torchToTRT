import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

# Load the ONNX model
model = onnx.load("unet.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)

rep = backend.prepare(model, device="CUDA:0")  # or "CPU"

outputs = rep.run(np.random.randn(1, 1, 572, 572).astype(np.float32))

print(outputs[0].shape)
