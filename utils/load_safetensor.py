from safetensors.torch import load_file, save_file
import torch

tensors = load_file("/home/lc/cv_models/Qwen2-0.5B/model.safetensors")

# for key, tensor in tensors.items():
#     print(f"Tensor name: {key}, Tensor shape: {tensor.shape}")

import base64

def text_to_base64_number(text):
    # 将文本编码为base64
    encoded_bytes = base64.b64encode(text.encode('utf-8'))
    
    # 将base64编码的bytes转换为字符串
    encoded_str = encoded_bytes.decode('utf-8')
    
    # 将base64字符串转换为数字
    # 这里我们简单地将base64字符串视为一个数字的base-64表示
    base64_number = int(encoded_str, 36)  # base64的字符集大小为36 (A-Z, a-z, 0-9, +, /)
    
    return base64_number

# 使用示例
text = "一段文本"
number = text_to_base64_number(text)
print(f"The base64 number for the text is: {number}")

# 假设我们有一个Base64编码的字符串
# encoded_str = 'SGVsbG8gV29ybGQh'  # 这是"Hello World!"的Base64编码

# 使用base64.b64decode()函数进行解码
decoded_bytes = base64.b64decode(encoded_str)

# 将解码后的字节串转换为普通字符串（假设它是UTF-8编码的文本）
decoded_str = decoded_bytes.decode('utf-8')

print(decoded_str)  # 输出：Hello World!


import base64

# 原始数据（必须是字节类型）
original_data = b'Hello, World!'

# 进行Base64编码
encoded_data = base64.b64encode(original_data)

# Base64编码后的数据是字节类型，可以打印或用于其他用途
print(encoded_data)  # 输出可能是 b'SGVsbG8sIFdvcmxkIQ=='

# 假设encoded_data是我们之前得到的Base64编码的字节串
# 进行Base64解码
decoded_data = base64.b64decode(encoded_data)

# 将解码后的字节串转换为普通字符串（假设原始数据是UTF-8编码的文本）
decoded_str = decoded_data.decode('utf-8')

print(decoded_str)  # 输出：Hello, World!