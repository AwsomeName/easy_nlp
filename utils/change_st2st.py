import torch
import safetensors.torch

from safetensors.torch import load_file, save_file
import torch

from_path = "/hy-tmp/Qwen2.5-0.5B-Instruct/model.safetensors"
safetensors_path = "/hy-tmp/YJT2.5-0.5B-Instruct/model.safetensors"

tensors_dict = load_file(from_path)

for key in tensors_dict:
    print("Key:", key)
    # print(tensors_dict[key])

# def convert_bin_to_safetensors(bin_path, safetensors_path):
#     # 加载bin文件
#     state_dict = torch.load(bin_path, map_location="cpu")
    
safetensors.torch.save_file(tensors_dict, safetensors_path)
    


# convert_bin_to_safetensors(bin_path, safetensors_path)
