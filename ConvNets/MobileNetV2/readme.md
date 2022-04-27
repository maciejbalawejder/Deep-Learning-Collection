# MobileNet
MobileNet was introducted in 2017 paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). The main goal was to imporve the efficiency of Deep Learning models to bring them on mobile devices. They achieved by introducting two things:
#### 1) __Depthwise Separable Convolution__ :

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNet/depthwiseblock.png"
>
</p>

Which consists of two operations:
- __Depthwise Convolution__ - it's extreme version of group convolution with __3x3__ kernel, where number of input channels is equal to the number of groups.
    
      depthwise = Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=3, stride=1, padding=1, groups=n_in) 
    
- __Pointwise Convolution__ - it's simple __1x1__ Conv that generates new features

      pointwise = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0)

#### 2) __Two additional hyperparamters to create smaller and faster models__ :
      
      α = (0, 1] - width multiplayer, it modifies input(α * M) and output(α * N) channels.
      
      ρ = (0, 1] - resolution multiplayer, it modifies input resolution (ρ * 224) and internal representations.
      
      Baseline configuration is α = 1 and ρ = 1.

# Architecture

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNet/block.png"
>
</p>

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/MobileNet/architecture.png"
>
</p>
