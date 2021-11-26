!python -m spacy download en
!python -m spacy download de

import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def save_checkpoint(state, path, name):
    torch.save(state, path + name + '.pth')

def i2w(tensor, vocab, target=False):
    batch = [] # list of all sentances
    for sentence in range(tensor.shape[1]):
        sen = [] # list of one sentance converted from tensor with indexes to list with words
        for idx in tensor[1:,sentence]:
            word = vocab.vocab.itos[int(idx)]
            if word != '<eos>':
                sen.append(word)
            else:
                break
        if target:
            batch.append([sen]) # to calculate bleu score the target sentance needs to be in list
        else:
            batch.append(sen)
    return batch

def list2sentance(arr):
    sen = ""
    for word in arr:
        sen += ' ' + word
    sen = sen[1:]
    sen = sen.capitalize() # capitalize first letter 
    return sen


def datasetGenerator(batch, device, max_size=10_000):
    # Spacy_language => contains specific set of rules governing certain language and its specific in tokenization, stop words or functions. 
    spacy_eng = spacy.load('en')
    spacy_de = spacy.load('de')

    # Tokenizing => breaking the sentances into list of words
    def tokenize_eng(text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    # Creating Field which converts data to tensor and overall preprocess the data like a torchvision for images
    english = Field(tokenize = tokenize_eng, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)
    
    german = Field(tokenize = tokenize_de, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)


    # Splitting the data
    train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(german, english))
    
    # Creating vocabulary => list of words occuring in the dataset, min_freq => word needs to be used at least 2 times
    english.build_vocab(train_data, max_size=max_size, min_freq=2)
    german.build_vocab(train_data, max_size=max_size, min_freq=2)

    # Creating iterators over dataset => tensors
    train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
        (train_data, validation_data,test_data),
        batch_size=batch,
        device=device
    )
    return (train_iterator, validation_iterator, test_iterator), (train_data, validation_data, test_data), (english, german)
  
  
if __name__ == '__main__':
    (ti, vi, tei), _, (eng, ger) = datasetGenerator(128, 'cpu', max_size=10_000)
    test = next(iter(ti))
    src = i2w(test.trg, eng)
    for i in range(5):
        print(list2sentance(src[i]))    
