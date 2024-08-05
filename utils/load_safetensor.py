from safetensors.torch import load_file, save_file
import torch

tensors = load_file("model.safetensors")

for key, tensor in tensors.items():
    print(f"Tensor name: {key}, Tensor shape: {tensor.shape}")