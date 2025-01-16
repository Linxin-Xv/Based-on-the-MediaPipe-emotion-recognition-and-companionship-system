import torch

# 创建一个张量并尝试移动到GPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")
    
    # 尝试在GPU上创建张量
    x = torch.rand(3,3).to(device)
    print("张量设备位置:", x.device)
    
except Exception as e:
    print(f"错误: {str(e)}")
import os
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))