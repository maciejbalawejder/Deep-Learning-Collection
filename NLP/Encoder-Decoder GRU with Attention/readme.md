# Encoder-Decoder with Attention 
[Cho et al.]() and [Sutskever et al.]() presented the encoder-decoder architecture that could encode __variable-length vector__(e.g sentance in English with 10 words) into vector __fixed length vector__(context vector)
and decode it into __variable-length vector__(e.g German sentance with 6 words).

This approach is based on the bottleneck representation, which contains information from all the previous state, but when the sequence is long enough, initial informations are getting discarded and lost.

[Neural Machine Translation](https://arxiv.org/pdf/1409.0473.pdf) paper by Cho et al. is addresing this problem by creating __attention mechanism__, where context vector is calculated specifically for each output timestep. 

### Architecture:

<p align="center">
    <img src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Encoder-Decoder%20GRU%20with%20Attention/Example-of-Attention.png">
</p>

__Encoder:__
- takes __x<sub>T</sub>__ sequence
- bidirectional recurrent cell(GRU), which helps to keep the information around the word, not only the next word  
- all the hidden states(__ht__) are saved


__Context vector(c<sub>t</sub>)__:
- calculated and feeded for each cell in the Decoder 
- sum of all __a<sub>tT</sub>__

__Decoder:__
- outputs __y<sub>T</sub>__ sequence
- regular recurrent cell(GRU)
- takes the last hidden state from Encoder(it's not done in original paper, but it improves the performance)


### Attention mechanism:
At each timestep(t), we are calculating how similar is decoder state(__s<sub>t</sub>__) in relation the hidden state(__meaning__) of certain word in encoder. The similarity is an output of fully connected layer that is trained along with the encoder and decoder. Alright let put some maths into it. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;e_{t}&space;=&space;attention(s_{t-1},&space;h)&space;\\&space;\alpha_{t}&space;=&space;softmax(e_{t})&space;\\&space;c_{t}&space;=&space;\alpha_{t}&space;@&space;h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;e_{t}&space;=&space;attention(s_{t-1},&space;h)&space;\\&space;\alpha_{t}&space;=&space;softmax(e_{t})&space;\\&space;c_{t}&space;=&space;\alpha_{t}&space;@&space;h" title="\\ e_{t} = attention(s_{t-1}, h) \\ \alpha_{t} = softmax(e_{t}) \\ c_{t} = \alpha_{t} @ h" /></a>

- e<sub>t</sub> - the result of the attention layer
- alpha<sub>t</sub> - weights, gives the probability distribution which hidden states pay attention to

### Tricks and tips
I leave also leave few tricks for the implementation that might make it easier for you to understand: 
- the hidden state of bidrectional cell has shape [seq length, batch, 2 x hidden_size]
- to fit the __last decoder state__ [1, batch, 2 x hidden_size] as __first encoder state__ [1, batch, hidden_size], use __fully connected layer__ with input : 2 x hidden_size, and output : hidden_size
- use torch.bmm to perform the matrix-matrix product(__@__) of __weights__(alpha) and __hidden states__(h)
- GRU cell in PyTorch returns the last hidden state from all layers, pick the output from the last two

```
        hts, hidden = self.RNN(xt)
        hidden = torch.cat((hidden[-2], hidden[-1]),dim=1).unsqueeze(0)
        # [1,128,HIDDEN_SIZE * 2]    
```

- Attention layer input is concatenated [__s<sub>t-1</sub>__, __h__]


