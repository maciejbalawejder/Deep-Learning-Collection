# GoogLeNet(InceptionV1)
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=r92siBwTI8U&t=525s).*

GoogLeNet is a Deep CNN introduced in 2014 in a  ["Going deeper with convolutions"](https://arxiv.org/pdf/1409.4842.pdf) paper. The model achieved __first place__ in classification 2014 __ImageNet competition__. The key takeaways from the article are:
- Increased __depth__ and __width__ of CNN can increse the processing performance and final result 
- Smaller __filters(1x1)__ are great to control the number of channels

# Usage
```python
import torch
from googlenet_pytorch import GoogLeNet
googlenet = GoogLeNet() # default in_channels=3 and output classes=1000
image = torch.rand(1, 3, 224, 224)
outputs = googlenet(image) # List of three outputs [aux_classifier_1, aux_classifier_2, classifier]
for out in outputs:
    print(out.shape)
```

# Architecture

![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/GoogLeNet/architecture%20table.png)
![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/GoogLeNet/architecture%20graph.png)



