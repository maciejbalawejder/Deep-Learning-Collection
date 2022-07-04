# Transfomer
Transformer the bread-and-butter of current AI, the horsepower of the biggest breakthrough models such as __GPT-3__ or __AlphaFold__. It was introducted in 2017 in famous ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf) paper. 

Previously the NLP was dominated by RNNs with Encoder-Decoder structure(+Attention) which work well on analysing the relationship between close words, but lose its accuracy on longer paragrath or articles, and they were computationally expensive. 

In 2016, Convolution Neural Networks came to play with [ByteNet(WaveNet)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), where the whole sentance was input at once. The model provided SOTA results in translation, but the relationships between layers were predfined, and it's actually better when these relations depend on the context of the words. 

So eventually __Transformer__ is a combination of Attention mechansim from RNNs and parrael computation capabilities of CNNs. 


# Usage
```python
    from model import Transformer, TransformerConfig

    src_vocab_size = 100 # size of source vocabulary
    trg_vocab_size = 50 # size of target vocabulary
    trg_pad = 1 # index for pad token in target vocabulary
    src_pad = 1 # index for pad token in source vocabulary
    src = torch.randint(0, src_vocab_size, (4, 1)) # dummy source and vocab sentance
    trg = torch.randint(0, trg_vocab_size, (2, 1))
    
    transformer = Transformer(
            TransformerConfig,
            src_vocab_size = src_vocab_size,
            trg_vocab_size = trg_vocab_size,
            src_pad = src_pad,
            trg_pad = trg_pad
            )

    outputs = transformer(src, trg) # [seq_len, batch, trg_vocab_size]

```

# Architecture
Architecture is build upon Encoder-Decoder stucture, but a little bit different than the previous models. 

![image](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/Transformer/images/transformerE-D.png)
