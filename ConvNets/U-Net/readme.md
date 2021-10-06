# U-Net

The U-Net model was introducted in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox with primary focus on ISBI cell tracking challenge, which they won by huge margin. During that time training model required many labeled data that is hard to obtain in medical environment. The encoder-decoder structure with skip-connections is enabling the features extraced in contracting path included in reconstruction. 
The main achievemtents of this novel approach were:
- using relatively small data to train the model with high accuracy
- outperform CNN models in __segmentation tasks__
- faster training time

### Training dataset 


### Hyperparameters

| Hyperparameter | Value | 
|:--------------:|:-----:|
| Loss function | Cross Entropy | 
| Optimizer    | Adam | 
| Learning rate | 0.0001 | 
| Batch   | 32 | 
| Epochs | 10 |


