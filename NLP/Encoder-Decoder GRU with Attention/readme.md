# Encoder-Decoder with Attention 
[Cho et al.]() and [Sutskever et al.]() presented the encoder-decoder architecture that could encode __variable-length vector__(e.g sentance in English with 10 words) into vector __fixed length vector(vector representation)__
and decode it into __variable-length vector__(e.g German sentance with 6 words).

This approach is based on the bottleneck representation, which contains information from all the previous state, but on the way there much information can be discarded and lost, this is why long sequences are lost in translation. 

The more human approach of understanding language is presented in [Neural Machine Translation](https://arxiv.org/pdf/1409.0473.pdf) paper by Cho et al., where __attention mechanism__ is choosing which words are important in the current context.  

### Architecture:

<p align="center">
    <img src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Encoder-Decoder%20GRU%20with%20Attention/Example-of-Attention.png">
</p>

__Encoder:__
- takes __x<sub>T</sub>__ sequence
- bidirectional recurrent cell(GRU), which helps to keep the information around the word, not only the next word  
- all the hidden states(__ht__) are saved

__Decoder:__
- outputs __y<sub>T</sub>__ sequence
- regular recurrent cell(GRU)
- takes the last hidden state from Encoder(it's not done in original paper, but it improves the performance)

__Context vector(c<sub>t</sub>)__:
- calculated and feeded for each cell in the Decoder 
- sum of all __a<sub>tT</sub>__( __att__ ension)

### Attention mechanism:
It is based on the context vector computed for each hidden cell in Decoder. 
