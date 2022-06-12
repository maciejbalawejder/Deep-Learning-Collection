# MobileNetV3

*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=0oqs-inp7sA&t=1363s).*

MobileNetv3 was introducted in 2019 paper [Searching for MobileNetV3
](https://arxiv.org/pdf/1905.02244.pdf). The main goal was a further imporvement of the efficiency of Deep Learning models on mobile devices. The new `bneck` block is 
based on the *inverted residual block* from __MobileNetv2__ but they also add the __Sqeeze-and-Excitation module__ on the top of it.
# Usage

```python
import torch
from mobilenetv3_pytorch import MobileNetv3

model_size = "small"
mobilenetv3 = MobileNetv3(model_size)

image = torch.rand(1, 3, 224, 224)
outputs = mobilenetv3(image) # [1, n_classes]
```

# BNeck block
<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV3/images/bneck.png"
>
</p>

- They new block looks as follows:
```
bneck = Sequential(
  expansion = ConvBlock(n_in, n_exp, 1, 1, act)
  depthwise = ConvBlock(n_exp, n_exp, 1, 1, act, groups=n_exp)
  se = SeBlock(n_exp)
  pointwise = ConvBlock(n_exp, n_out, 1, 1, Identity())
  )
```
- Another improvement is using different activation function : `ReLU` in shallow layers and `SiLU` in deeper parts, which helps to prevent dead neurons.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV3/images/nls.png"
>
</p>

# Architecture
Unlike its precursors, researchers were using the __Natural Architecture Search(NAS)__ and __NetAdapt__ to find the optimal architecture. 
They also manually tailored the first convolution and last stage of the network, reducing the complexity of the model while maintaining the accuracy. 
Essentially, they presented two configurations of __MobileNetv3__:
### Large

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV3/images/large.png"
>
</p>

### Small

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV3/images/small.png"
>
</p>
