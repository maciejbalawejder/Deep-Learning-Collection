# EfficientNet
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=eFMmqjDbcvw&t=108s).*


EfficientNet was introducted in 2019 paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf). 
The main goal was to improve the efficiency of Deep Learning models to bring them on mobile devices. The main meat of the paper is introducting the __compound scalling method__. 
The parameter `Φ` denotes available computational resources. Based on this values the size of the network and input resolution is adjusted.

# Usage
```python
import torch
from efficientnet_pytorch import EfficientNet

model_config = "B0" # ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"] -> are available
efficientnet = EfficientNet(model_config)

image = torch.rand(1, 3, 224, 224)
outputs = efficientnet(image) # [1, n_classes]
```

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/EfficientNet/scalling.png"
>
</p>


<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/EfficientNet/coefficients.png"
>
</p>



# Architecture

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/EfficientNet/squeeze&excitaionMobileV3.png"
>
</p>

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/EfficientNet/baseline.png"
>
</p>

# To-do:
- [x] add Stochastic Depth
