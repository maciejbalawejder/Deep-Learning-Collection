# AlexNet

*You can find the accompanying video [here](https://www.youtube.com/watch?v=3uNu5x2wZ5s&t=817s).*


__AlexNet__ was introduced in 2012 in [ImageNet Classification with Deep Convolution Neural Network](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) paper. It was a breakthrough model that started the Deep Learning revolution. The main __takeaways__ from the article are: 
- Large CNN's __outperform__ other techniques significantly on large datasets like ImageNet
- CNN's utilize the __GPU__
- __ReLU__ speeds up computation significantly
- __Overfitting__ can be combated using different techniques like:
  * Dropout
  * Data augmentation


# Usage
```python
import torch
from alexnet_pytorch import AlexNet

alexnet = AlexNet() # in_channels = 3, classes = 1000 as default

image = torch.rand(1, 3, 224, 224)
outputs = alexnet(image) # [1, n_classes]

```

# Architecture
![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/AlexNet/architecture.png)
![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/AlexNet/volumes.png)

