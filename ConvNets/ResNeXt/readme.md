# ResNeXt
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=CANodHhCyCw&t=395s).*

ResNeXt model was introduced in a 2016 paper ["Aggregated Residual Transformations for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431.pdf) by Kaming He et al. This novel architecture is a combination of __VGG/ResNet__ way of stacking more layers, and __Inception's__ *split-transform-merge* technique. The ultimate goal they are tackling is getting __better results__ without __increasing depth__(more computation).

They introduce a new parameter, *cardinality*(C), which controls the bottleneck width, and prove that it's more important than depth and width(feature maps). They also used __group convolution__ in their bottleneck block that speeds up computation and yields the same results as *1x1* projections.

# Usage
```python
import torch
from resnext_pytorch import ResNeXt

config_name = 50 # 101 and 150 are also available
C = 32 # cardinality
resnext50 = ResNeXt(config_name, in_channels=3, classes=1000, C=C)

image = torch.rand(1, 3, 224, 224)
outputs = resnext50(image) # [1, n_classes]
```

# Architecture

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/ResNeXt/block.png"
>
</p>

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/ResNeXt/architecture.png"
>
</p>

