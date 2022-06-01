# MobileNetV2
MobileNetv2 was introducted in 2018 paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks
](https://arxiv.org/pdf/1801.04381.pdf). The main goal was a further imporvement of the efficiency of Deep Learning models on mobile devices. The new architecture is based on the *inverted residual block* similar to the ResNet bottleneck but they remove the final ReLU which helps to prevent infromation loss.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV2/invertedblock.png"
>
</p>

Which is based on the `narrow -> wide -> narrow` structure unlike original bottleneck hence why it's called inverted:
- __Expansion Convolution__ - the __1x1 Conv2d__ that generates larger feature space
      
      expansion = Conv2d(in_channels=n_in, out_channels=n_exp, kernel_size=1, stride=1, padding=0, act=ReLU) 

- __Depthwise Convolution__ - it's extreme version of group convolution with __3x3__ kernel, where number of input channels is equal to the number of groups.
    
      depthwise = Conv2d(in_channels=n_exp, out_channels=n_exp, kernel_size=3, stride=1, padding=1, groups=n_in, act=ReLU) 
    
- __Pointwise Convolution__ - it's simple __1x1__ Conv __w/o activation__ that squeeze expansion channels to the input channels 

      pointwise = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, act=Identity)


# Architecture

#### __There are two additional hyperparamters to create smaller or larger models__ :
      
      α = (0, 1.4] - width multiplier, it modifies input(α * M) and output(α * N) channels.
      
      ρ = (0, 1.4] - resolution multiplier, it modifies input resolution (ρ * 224) and internal representations.
      
      Baseline configuration is α = 1 and ρ = 1.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV2/block.png"
>
</p>

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNetV2/architecture.png"
>
</p>
