import torch
from torch import Tensor
import torch.nn as nn 

class StochasticDepth(nn.Module):
    """Stochastic Depth / Drop Path module

    Parameters
    ----------
        p : float
            stochastic depth probablity

    """
    
    def __init__(
        self,
        p : float = 0.5
        ):
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
            ret : torch.Tensor
                Output tensor of shape (batch_size, out_channels, height, width).

        """

        mask_shape = (x.shape[0],) + (1,)*(x.ndim - 1) # mask shape: [batch, 1, 1, 1]
        mask = torch.empty(mask_shape).bernoulli_(self.p) / self.p
        
        if self.training: 
            x = mask * x

        return x 

