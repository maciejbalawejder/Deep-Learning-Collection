# Encoder - Decoder LSTM

The model encoder-decoder RNN was introduced by [Cho et al.](https://arxiv.org/pdf/1406.1078.pdf) and [Sutskever et al.](https://arxiv.org/pdf/1409.3215.pdf). The main goal was to map two sequences of different lengths. It is especially important in __translation__, where for example sentance in English can have 10 words, but when it is translated to German it only has 6. This papers proved that it is possible and this approach can surpass the previous state-of-the-art performance level. 
### Architecture:

![Model](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Encoder-Decoder%20LSTM/imgs/model.jpeg)

-----

__Encoder__:
- stack of several recurrent units (RNN, LSTM, GRU)
- xi represents each word
- ht hidden state at given timestep

__Encoder vector__:
- final hidden state of encoder and intitial hidden state for decoder
- representation of essential information from input

__Decoder__: 
- stack of several recurrent units 
- each units produce a word
- end of the sequence is tokenized with word "EOS"

### Dataset:
Multi30k - translation from German to English

```
German : Eine gruppe von mannern ladt baumwolle auf einen lastwagen.
English : A group of men are loading cotton onto a truck.
```

```
German : Ein mann schlaft in einem grunen raum auf einem sofa. 
English : A man sleeping in a green room on a couch.
```
### __Hyperparameters__:
```
BATCH = 128
LEARNING_RATE = 0.001
LAYERS = 1
HIDDEN_DIM = 512
EPOCHS = 25
EMBEDDING_DIM = 300
P = 0.5 # Dropout rate
```

### Results
#### 1) Loss
![Loss](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Encoder-Decoder%20LSTM/imgs/loss.png) 
#### 2) Bleu score = 16.42 = ["Hard to get the gist"](https://cloud.google.com/translate/automl/docs/evaluate)
![Bleu](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Encoder-Decoder%20LSTM/imgs/bleu.png)
#### 3) Translations example
```
Target : Woman in a pink shirt and wearing glasses.
Prediction : Woman with glasses in a a shirt.
```
```
Target : A young lady doing yoga on the beach.
Prediction : A young lady is on the the beach.
```
```
Target : A woman wearing a hat with making bread.
Prediction : A woman in a hat hat. 🙃

```
### Improvements
- [ ] train on more data
- [ ] GRU cells
- [ ] Beam serach instead of Force Teaching
