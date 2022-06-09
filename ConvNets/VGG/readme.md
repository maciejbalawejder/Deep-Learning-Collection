# VGG
*You can find the accompanying paper review video [here](https://www.youtube.com/watch?v=c1lqOpFCJkw&t=478s).*

VGG is a Deep CNN introduced in 2014 in a  ["Very Deep Convolutional Network for Large Scale Image Recognition"](https://arxiv.org/abs/1409.1556) paper. The model achieved __first place__ in localization and __second__ in classification in the 2014 __ImageNet competition__. The key takeaways from the article are:
- Increased __depth__ of CNN is beneficial for accuracy
- Smaller __filters(3x3)__ yields a better result
- Interesting __cropping and scaling__ techniques that increase the dataset(scale jittering)
- The network can __generalize__ well on different datasets

# Architecture
They tried __six__ different architectures, with 11, 13, 16, and 19 layers. I haven't included configuration __C__ in the code since 16 layers with __1x1 filters__ performed worse than one with __3x3__. Also, configuration __A-LRN__ contained a __local normalization layer__ that didn't improve the performance, so I skipped it.


![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/VGG/architectures.png)
![](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/VGG/results.png)
