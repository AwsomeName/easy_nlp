import torch
import safetensors.torch

def convert_bin_to_safetensors(bin_path, safetensors_path):
    # 加载bin文件
    state_dict = torch.load(bin_path, map_location="cpu")
    
    safetensors.torch.save_file(state_dict, safetensors_path)
    
bin_path = ""
safetensors_path = ""

convert_bin_to_safetensors(bin_path, safetensors_path)
