import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from config import get_weights, load_weights
import wget
import numpy as np

class Embeddings(nn.Module):
    """ Patch and Position Embeddings + CLS Token. 

    Parameters
    ----------
    in_channels : int - input channels of the image
    patch_size : int - size of the patch
    img_size : int - height or width of the image assuming that it's square
    d_size : int - embedding dimension from config class
    p_emb : float - embedding dropout rate

    Attributes
    ----------
    projection : nn.Conv2d - projection of input image to the embedding patches 
    n_patches : int - number of patches
    positions : nn.Parameter - position embeddings randomly intialized at the start
    cls_token : nn.Parameter - learnable classification token(totally unnecessary)
    emb_drop : nn.Dropout - embedding dropout
    
    """

    def __init__(
            self,
            in_channels,
            patch_size,
            img_size,
            d_size,
            p_emb
            ):

            super().__init__()
            
            self.projection = nn.Conv2d(
                in_channels = in_channels,
                out_channels = d_size,
                kernel_size = patch_size,
                stride = patch_size
            )
            
            self.n_patches = (img_size // patch_size)**2
            self.positions = nn.Parameter(data = torch.zeros(size=(1, d_size, self.n_patches + 1)))
            self.cls_token = nn.Parameter(data = torch.zeros(size=(1, d_size, 1)))
            self.emb_drop = nn.Dropout(p=p_emb)
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + 1, d_size)
        """

        batch, in_channels, height, width = x.shape
        patch_emb = self.projection(x).flatten(2) # shape : (batch, d_size, n_patches)
        patch_emb = torch.cat((self.cls_token.expand(batch, -1, -1), patch_emb), axis=-1) # shape : (batch, d_size, n_patches + 1)
        pos_emb = patch_emb + self.positions
        return self.emb_drop(pos_emb.permute(0, 2, 1))

class MultiHeadAttention(nn.Module):
    """ Attention mechanism.

    Parameters
    ----------
    d_size : int - embedding dimension from config class
    n_heads : int - number of heads 
    p_att : float - attention dropout rate

    Attributes
    ----------
    Q : nn.Linear - query projection 
    V : nn.Linear - key projection
    K : nn.Linear - value projection
    linear : nn.Linear - final projection in attention mechanism
    att_drop : nn.Dropout - attention dropout layer
    
    """

    def __init__(
        self,
        d_size,
        n_heads,
        p_att
        ):

        super().__init__()

        self.linear = nn.Linear(d_size, d_size)
        self.att_drop = nn.Dropout(p=p_att)
        self.n_heads = n_heads
        self.head_dim = d_size // n_heads

        self.Q = nn.Linear(d_size, d_size)
        self.V = nn.Linear(d_size, d_size)
        self.K = nn.Linear(d_size, d_size)

        
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)
        """

        batch, n_patches, d_size = x.shape
        mask = torch.triu(input = torch.ones((n_patches, n_patches))).expand(batch, 1, n_patches, n_patches)


        q = self.Q(x).reshape(batch, n_patches, self.n_heads, self.head_dim)
        k = self.K(x).reshape(batch, n_patches, self.n_heads, self.head_dim)
        v = self.V(x).reshape(batch, n_patches, self.n_heads, self.head_dim)

        QK = torch.einsum("bqhd, bkhd -> bhqk", [q, k]) / math.sqrt(d_size)
        QK = QK.masked_fill(mask==0, torch.finfo(torch.float32).min)

        scores = self.att_drop(F.softmax(QK, dim=3))
        output = torch.einsum("bhqk, bvhd -> bqhd", [scores, v])

        concat = output.reshape(batch, n_patches, d_size)
        return self.att_drop(self.linear(concat))
        
class MLP(nn.Module):
    """ Feed Forward module. 

    Parameters
    ----------
    d_size : int - embedding dimension from config class
    mlp_size : int - expansion dimension in mlp module
    p_mlp : float - mlp dropout rate

    Attributes
    ----------
    ff : nn.Sequential - all layers in one module
    """

    def __init__(
        self,
        d_size,
        mlp_size,
        p_mlp
        ):

        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, d_size),
            nn.Dropout(p_mlp)
        )
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)
        """

        return self.ff(x)

class ViTBlock(nn.Module):
    """ ViT Block with Multi-Head Attention module, MLP and Layer Norms. 

    Parameters
    ----------
    d_size : int - embedding dimension from config class
    n_heads : int - number of heads
    mlp_size : int - expansion dimension in mlp module
    p_att : float - attention dropout rate
    p_mlp : float - mlp dropout rate
    eps : float - a value added in denominator of Layer Norm for numerical stability

    Attributes
    ----------
    mha : nn.Module - Multi-Head Attention module
    mlp : nn.Module - MLP module
    ln1 : nn.LayerNorm - layer normalization 1
    ln2 : nn.LayerNorm - layer normalization 2
    """

    def __init__(
        self,
        d_size,
        n_heads,
        mlp_size,
        p_att,
        p_mlp,
        eps = 1e-6
        ):

        super().__init__()

        self.mha = MultiHeadAttention(
            d_size = d_size,
            n_heads = n_heads,
            p_att = p_att
        )

        self.mlp = MLP(
            d_size = d_size,
            mlp_size = mlp_size,
            p_mlp = p_mlp
        )

        self.ln1 = nn.LayerNorm(
            normalized_shape=d_size,
            eps = eps
        )

        self.ln2 = nn.LayerNorm(
            normalized_shape=d_size,
            eps = eps
        )
    
    def forward(self, x):
        """ Forward function.
        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)
        """

        x = x + self.mha(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class VisionTransformer(nn.Module):
    """ ViT architecture with Embeddings, ViTBlocks and Classification head. 

    Parameters
    ----------
    config : class - configuration class with model specifications
    pretrained : bool - defines whether model was pretrained on ImageNet21k
    fine_tuned : bool - defines whether model was fine-tuned on ImageNet1k after pretraining.
    pre_logits : bool - defines wheter model has layer before the classification head.
    in_channels : int - number of input channels, default 3.
    
    Attributes
    ----------
    encoder_norm : nn.LayerNorm - layer normalization before classification head
    pre_logits_layer : nn.Linear - last linear projection before classification head
    embeddings : nn.Module - patch and positional embeddings with cls token
    vit_blocks : nn.ModuleList - collection of vit blocks
    head : nn.Linear - classification head
    """

    def __init__(
        self,
        config,
        pretrained = False,
        fine_tuned = False,
        pre_logits = False,
        in_channels = 3
        ):

        super().__init__()

        self.pre_logits = pre_logits

        self.encoder_norm = nn.LayerNorm(
            normalized_shape = config.d_size,
            eps = config.eps
        )
        
        if self.pre_logits:
            self.pre_logits_layer = nn.Linear(config.d_size, config.d_size)

        self.embeddings = Embeddings(
            in_channels = in_channels,
            patch_size = config.patch_size,
            img_size = config.img_size,
            d_size = config.d_size,
            p_emb = config.p_emb
        )

        self.vit_blocks = nn.ModuleList([
            ViTBlock(
                d_size = config.d_size,
                n_heads = config.n_heads,
                mlp_size = config.mlp_size,
                p_att = config.p_att,
                p_mlp = config.p_mlp
            ) for i in range(config.layers)
        ])

        self.head = nn.Linear(config.d_size, config.out_channels)
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Outputs
        -------
        Tensor - with shape (batch, out_channels)
        """
        x = self.embeddings(x)
        for block in self.vit_blocks:
            x = block(x) # shape : [batch, n_patches, d_size]
        
        x = x[:, 0, :] # shape : [batch, d_size] - we are taking only the cls token

        if self.pre_logits:
            x = self.pre_logits_layer(x)
            x = torch.tanh(x)

        return self.head(self.encoder_norm(x))

class ViT(nn.Module):
    """ Whole ViT architecture with additional options to load the pretrained and fine-tuned weights. 
    
    Parameters
    ----------
    config_name : str - configuration name "B-16", "B-32", "L-16", "L-32"
    pretrained : bool - defines whether model was pretrained on ImageNet21k
    fine_tuned : bool - defines whether model was fine-tuned on ImageNet1k after pretraining.
    in_channels : int - number of input channels, default 3.
    
    
    Attributes
    ----------
    vit_model : nn.Module - the Visual Transformer model

    """

    def __init__(
        self,
        config_name,
        pretrained=False,
        fine_tuned=False,
        in_channels=3
        ):

        super().__init__()

        config, pre_logits, torch_weights = get_weights(config_name, pretrained, fine_tuned)
        self.vit_model = VisionTransformer(config, pretrained, fine_tuned, pre_logits)
        
        if torch_weights != 0:
            self.vit_model = load_weights(torch_weights, self.vit_model)


    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Outputs
        -------
        Tensor - with shape (batch, out_channels)
        """
        return self.vit_model(x)
    
# Sanity checks
if __name__ == "__main__":
    img = torch.rand(1, 3, 224, 224)
    vit = ViT("B-16", False, False, False) 
    print(vit(img).shape)