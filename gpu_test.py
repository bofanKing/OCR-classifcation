import torch
print("是否有 GPU：", torch.cuda.is_available())
print("GPU 名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")


import numpy as np
print(np.array([1, 2, 3]))
