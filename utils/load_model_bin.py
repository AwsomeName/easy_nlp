import torch
    
bin_path = ""
state_dict = torch.load(bin_path, map_location="cpu")

for key in state_dict:
    print(key)
