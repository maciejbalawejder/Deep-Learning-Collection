# Transfomer
Transformer the bread-and-butter of current AI, the horsepower of the biggest breakthrough models such as __GPT-3__, __AlphaFold__, or __AlphaStar__. Introducted in 2017 in famous ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf) paper. 

Previously the NLP was dominated by RNNs with Encoder-Decoder structure(+Attention) which work well on analysing the relationship between close words, but lose its accuracy on longer paragrath or articles, and they were computationally expensive. 

In 2016, Convolution Neural Networks came to play with [ByteNet(WaveNet)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), where the whole sentance was input at once. The model provided SOTA results in translation, but the relationships between layers were predfined, and we rather decide what to pay attention to. 

So eventually __Transformer__ is a combination of Attention mechansim from RNNs and parrael computation capabilities of CNNs. 

# Architecture

![image](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Transformer/transformerE-D.png)

Architecture is build upon Encoder-Decoder stucture, but a little bit different than the previous models. 

So eventually we end up with 5 building blocks, which combined together create Transformer:
### 1) __Embeddings__  
Turning words into high dimensional space, which is bascially __feature representation__ of the word.
### 2) __Positional Embeddings__
Since we pass whole sentance at once, the model does not have idea about the order of the words. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Why is it important? 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~ "Work to live vs live to work"

So we can trained additional __Embedding layer__ to recognize the positions of the word, or add certain __sin and cos__ value to the embeddings, which makes it distinguishable for the model.

### 3) __Multi Head Attention__ 
The heart of the model, bulid on __self-attention__ mechanism.
![image](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Transformer/multihead.PNG)

__What are V, K, Q?__

As we get the source sentance, we create 3 representation of it, Queries, Keys, Values, using simple Linear layer. The paper itself doesn't say much about them, so I will use analogy to the RNN's attention mechanism.
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\\&space;e_{t}&space;=&space;attention(s_{t-1},&space;h)&space;\&space;\rightarrow&space;\&space;QK&space;=&space;attention(Q,&space;K)&space;\\&space;\alpha_{t}&space;=&space;softmax(e_{t})&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\rightarrow&space;score&space;=&space;softmax(\frac{QK}{\sqrt{d_{k}}})&space;\\&space;c_{t}&space;=&space;\sum&space;_{j}\alpha_{tj}h_{j}&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\rightarrow&space;c&space;=&space;\sum&space;_{j}score_{j}V_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\\&space;e_{t}&space;=&space;attention(s_{t-1},&space;h)&space;\&space;\rightarrow&space;\&space;QK&space;=&space;attention(Q,&space;K)&space;\\&space;\alpha_{t}&space;=&space;softmax(e_{t})&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\rightarrow&space;score&space;=&space;softmax(\frac{QK}{\sqrt{d_{k}}})&space;\\&space;c_{t}&space;=&space;\sum&space;_{j}\alpha_{tj}h_{j}&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\rightarrow&space;c&space;=&space;\sum&space;_{j}score_{j}V_{j}" title="\\ e_{t} = attention(s_{t-1}, h) \ \rightarrow \ QK = attention(Q, K) \\ \alpha_{t} = softmax(e_{t}) \ \ \ \ \ \ \ \rightarrow score = softmax(\frac{QK}{\sqrt{d_{k}}}) \\ c_{t} = \sum _{j}\alpha_{tj}h_{j} \ \ \ \ \ \ \ \ \ \ \ \ \rightarrow c = \sum _{j}score_{j}V_{j}" /></a>
</p>

In the encoder and Masked attention in Decoder we calculate __self-attention score__ between query and keys, and we see how words are related to each other. In this case __query__ is a different represenation of the incoming sentance. 
But when we get to the part of decoder where attention is calculated based on __encoder state__, query is an representation of the attention in the __target sentance__(output of Masked Attention). 

The attention score is calculated using __dot product__ instead of Linear layer, which is much more efficient. Also it's scaled using the square root of the embedding dimensions(__dk__) to not blow up.

__What is Multi-head attention?__

__Multi-head__ means we split by the embedding size query, key, value, and calculate attention score for each chunk("head"). Each head is like a feature detector that learns different semantic meanings of sentences, for example grammar, vocabulary, conjugation etc. 

__Why decoder part is masked?__




