# Implementation of LeNet-5 architecture

LeNet-5 was introducted in 1998 by Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner for handwritten and machine-printed character recognition. Even though the architecture is straight forward it outperformed any avaiable methos back then with 0.95% error on the test data. The model consists of two sets of convolutional and average pooling layers, followed by three fully connected layers. The paper __Gradient Based Recognition applied to document recognition__ is detalied in the architecture and in depth explanations, but little is sad about the choice of the hyperparameters. Here are the intial values:

| Hyperparameters | Value| 
|:---------------:|:----:|
| Batch           | 16   |
| Optimizer       | SGD  |
| Learning rate   | 0.01 |
| Epochs          | 10   |

 ![Architecure](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/LeNet/figures/architecture.png) 

### Optimizing hyperparameters 
- Grid search
