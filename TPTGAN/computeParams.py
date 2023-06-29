import torch
from new_model import Net
from thop import profile

model = Net()
inputtensor = torch.randn(1, 3, 128, 128)
flops, params = profile(model, inputs=(inputtensor,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
