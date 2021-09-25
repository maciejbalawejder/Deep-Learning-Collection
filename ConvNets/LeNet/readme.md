# Implementation of LeNet-5 architecture

LeNet-5 was introducted in 1998 by Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner for handwritten and machine-printed character recognition. Even though the architecture is straight forward it outperformed any available methods back then with only 0.95% error on the test data. 

The model consists of two sets of convolutional and average pooling layers, followed by three fully connected layers. 

The paper __Gradient Based Recognition applied to document recognition__ is detalied in the architecture and in depth explanations, but little is said about the choice of the hyperparameters. Here are the intial values I used:

| Hyperparameters | Value| 
|:---------------:|:----:|
| Batch           | 16   |
| Optimizer       | SGD  |
| Learning rate   | 0.1 |
| Epochs          | 10   |

 ![Architecure](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/LeNet/figures/architecture.png) 
 
 *I replace the Gaussian connection with the cross entropy loss function which achieve the same results more efficiently.

### Optimizing hyperparameters 
- Grid search
