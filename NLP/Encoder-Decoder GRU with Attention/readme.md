# Encoder-Decoder with Attention 
[Cho et al.]() and [Sutskever et al.]() presented the encoder-decoder architecture that could encode __variable-length vector__(e.g sentance in English with 10 words) into vector __fixed length vector(vector representation)__
and decode it into __variable-length vector__(e.g German sentance with 6 words). It achieved the state-of-the-art results in __Machine Translation__. 
These approach is based on the bottleneck representation, which contains information from all the previous states 
