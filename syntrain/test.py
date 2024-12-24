import torch

a = torch.tensor([[j for j in range(i, i+6)] for i in range(10)])
print(a)
index = torch.randperm(a.size(0))
a = a[index]
print(a)
    
