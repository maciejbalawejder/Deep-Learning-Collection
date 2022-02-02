# Implementation of LeNet-5 architecture

LeNet-5 was introducted in 1998 by Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner for handwritten and machine-printed character recognition. Even though the architecture is straight forward it outperformed any available methods back then with only 0.95% error on the test data. 

The paper [__Gradient Based Recognition applied to document recognition__](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) is detalied in the architecture and in depth explanations, but little is said about the choice of the hyperparameters. Here are the intial values I used:

| Hyperparameters | Value| 
|:---------------:|:----:|
| Batch           | 32   |
| Optimizer       | Adam  |
| Learning rate   | 0.00001 |
| Epochs          | 10   |

 ![Architecure](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/LeNet/figures/architecture.png) 
 
 *I replace the Gaussian connection with the cross entropy loss function which achieve the same results more efficiently.

### Results
- 97% accuracy on test data


![Loss function](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/ConvNets/LeNet/figures/Loss.png)
