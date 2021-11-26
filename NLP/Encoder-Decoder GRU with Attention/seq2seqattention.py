import torch
import random
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, layers, p):
        super(Encoder, self).__init__() 
        # Input size is source language vocabulary size       
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, 
                           num_layers=layers, dropout=p, bidirectional=True)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self,xt):
        # xt.shape: [SEQ_LEN,128]

        xt = self.dropout(self.embedding(xt))
        # xt shape:[SEQ_LEN,128,EMBEDDING_SIZE]

        hts, hidden = self.RNN(xt)
        # hidden shape: [LAYERS*2,128,HIDDEN_SIZE] => *2 because bidirectional
        # outputs shape: [SEQ_LEN,128,2*HIDDEN_SIZE] => all hidden states

        hidden = torch.cat((hidden[-2], hidden[-1]),dim=1).unsqueeze(0)
        # [1,128,HIDDEN_SIZE*2]

        hidden = torch.tanh(self.fc(hidden))
        # [1,128,HIDDEN_SIZE] Final hidden state that we want to fit into the decoder

        return hts, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, layers, p):
        super(Decoder, self).__init__()        
        # Output size and input size is target language vocabulary size

        self.embedding = nn.Embedding(input_size, embedding_size) 
        self.RNN = nn.GRU(input_size=embedding_size+hidden_size*2, hidden_size=hidden_size, 
                           num_layers=layers, dropout=p)
        self.dense = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(p=p)
        self.input_size = input_size
        self.attn = Attention(hidden_size*3)

    def forward(self, x, st, ht):
        x = x.unsqueeze(0)
        # x shape: [1, BATCH]
        # state shape: [1, BATCH, HIDDEN_SIZE]

        x = self.dropout(self.embedding(x))
        # x shape: [1, BATCH, EMBEDDING_SIZE]
        
        weights = self.attn(st, ht)
        ct = torch.bmm(weights,ht.permute(1,0,2))
        # ct shape: [BATCH, 1, HIDDEN_SIZE*2]

        ct = ct.permute(1, 0, 2)
        # ct shape: [1, BATCH, HIDDEN_SIZE*2]

        x = torch.cat([x,ct], dim=2)

        output, state = self.RNN(x, st)
        # output shape: [1, BATCH, HIDDEN_SIZE] output and the state are the same vectors
        # state shape: [1, BATCH, HIDDEN_SIZE]

        pred = self.dense(output)
        # pred shape: [1, BATCH, INPUT_SIZE]

        pred = pred.squeeze(0)
        # pred shape: [BATCH, INPUT_SIZE]
        return pred, state


class Attention(nn.Module):
    def __init__(self, input_size): 
        super(Attention, self).__init__()
        self.a = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, st, hiddens):
        # st shape: [1, BATCH, HIDDEN_DIM]
        # hiddens shape: [33, BATCH, HIDDEN_DIM*2] => 33 is a sequence length
        st = st.repeat(hiddens.shape[0], 1, 1)
        # st shape: [33, BATCH, HIDDEN_DIM]

        concat = torch.cat([st,hiddens], dim = 2)
        # concat shape: [33, BATCH, HIDDEN_DIM*3]

        energy = self.relu(self.a(concat)) # energy score- eij paper
        # energy shape: [33, BATCH, 1]

        weights = F.softmax(energy,dim=0) # weight vector alphaij in paper
        # weights shape: [33,128,1]

        return weights.permute(1,2,0)

        
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, force_ratio):
        # target, source shape: [SEQ_LEN, BATCH]
        target_len, batch = target.shape
        
        outputs = torch.zeros(target_len, batch, self.decoder.input_size)
        
        ht, st = self.encoder(source) # ht - all hidden states from decoder, st - last hidden state in decoder and first hidden state for encoder
        xt = target[0]

        for t in range(1,target_len):
            yt, st = self.decoder(xt, st, ht)
            outputs[t] = yt.unsqueeze(0)
            
            pred = yt.argmax(1)

            if random.random() > force_ratio:
              xt = pred
            else:
              xt = target[t]

        return outputs

if __name__ == "__main__":
  enc = Encoder(400,300,512,2,0.5)
  dec = Decoder(400,300,512,1,0.5)
  eout = enc(torch.rand([33,128]).long())
  dout = dec(torch.rand([128]).long(), eout[1], eout[0])
  seq = Seq2Seq(enc,dec)
  out = seq(torch.rand([33,128]).long(), torch.rand([28,128]).long(), 0)
  print(out.shape)
