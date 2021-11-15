Encoder - Decoder LSTM

The model encoder-decoder RNN was firstly introduced by Google in 2014. The main goal was to map two sequences of different lengths. It is especially importatnt in machine translation, where for example sentance in English can have 10 words, while it is translated to Spanish it only has 6.

Architecture

Encoder:
- stack of several recurrent units (RNN, LSTM, GRU)
- xi represents each word
- ht hidden state at given timestep

Encoder vector:
- final hidden state of encoder and intitial hidden state for decoder
- representation of essential information from input

Decoder: 
- stack of several recurrent units 
- each units produce a word
- end of the sequence is tokenized with word "EOS"

Dataset:
-----
Multi30k - translation from German to English
Examples:
'''
German : Eine gruppe von mannern ladt baumwolle auf einen lastwagen.
English : A group of men are loading cotton onto a truck.

'''

'''
German : Ein mann schlaft in einem grunen raum auf einem sofa. 
English : A man sleeping in a green room on a couch.

'''
`
