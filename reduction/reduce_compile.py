import torch

@torch.compile
def reduce_compile(x):
    return torch.sum(x)

reduce_compile(torch.randn(10).cuda())
