import torch
from torch import Tensor
import torch.nn as nn 

class StochasticDepth(nn.Module):
    " Stochastic Depth / Drop Path module which takes survival probability(p) and returns input with *dropped* examples in the batch. "
    
    def __init__(
        self,
        p : float = 0.5
        ):
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        mask_shape = (x.shape[0],) + (1,)*(x.ndim - 1)
        # mask shape: [batch, 1, 1, 1]
        mask = torch.empty(mask_shape).bernoulli_(self.p) / self.p
        
        if self.training : x = mask * x
        return x 

