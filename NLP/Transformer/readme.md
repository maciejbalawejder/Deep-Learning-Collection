# Transfomer
Transformer the bread-and-butter of current AI, the horsepower of the biggest breakthrough models such as __GPT-3__, __AlphaFold__, or __DeepStar__. Introducted in 2017 in famous ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf) paper. 

Previously the NLP was dominated by RNNs with Encoder-Decoder structure(+Attention) which work well on analysing the relationship between close words, but lose its accuracy on longer paragrath or articles, and they were computationally expensive. 

In 2016, Convolution Neural Networks came to play with [ByteNet(WaveNet)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), where the whole sentance was input at once. The model provided SOTA results in translation, but the relationships between layers were predfined, and we rather decide what to pay attention to. 

So eventually __Transformer__ is a combination of Attention mechansim from RNNs and parrael computation capabilities of CNNs. 

# Architecture

![image](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Transformer/imgs/e-dTransformer.png)


