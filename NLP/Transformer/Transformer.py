import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        # d_model = embedding_size

        positions = torch.arange(max_len).unsqueeze(1)
        # position shape: [5000, 1]

        PE = torch.zeros(max_len, 1, d_model)
        # PE shape : [5000, 1, 512]

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # div_term shape : [256]

        PE[:, 0, 0::2] = torch.sin(positions * div_term)
        PE[:, 0, 1::2] = torch.cos(positions * div_term)
        # 0::2 => [0,2,4...], 1::2 => [1,3,5...]

        self.register_buffer('PE', PE)  # save it as non-trainable parameter

    def forward(self, x):
        # x shape: [seq_len, batch, embedding_size]
        # PE shape : [max_len, 1, embedding_size]
        x = x + self.PE[:x.shape[0]] # we only want the sequence length [:33]

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, heads):
        super(MultiHeadAttention,self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = self.embedding_size // self.heads
        assert self.head_dim*heads == self.embedding_size

        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc = nn.Linear(self.embedding_size,self.embedding_size)

        self.softmax = nn.Softmax(dim=3)


    def forward(self, q, k, v, mask=None):
        batch = q.shape[1]
        query_len = q.shape[0]
        key_len = k.shape[0]
        value_len = v.shape[0]

        v = v.reshape(batch, self.heads, value_len, self.head_dim)
        q = q.reshape(batch, self.heads, query_len, self.head_dim)
        k = k.reshape(batch, self.heads, key_len, self.head_dim)

        V = self.V(v)
        Q = self.Q(q)
        K = self.K(k)
        # Q,K,V shapes: [batch, heads, len, head_dim]
        
        QK = torch.einsum('bhqd, bhkd->bhqk',[Q,K])
        # QK shape: [batch, heads, q_len, k_len] 

        if mask is not None:
            QK = QK.masked_fill(mask==0, float("-1e20"))

        filter = self.softmax(QK/(self.head_dim**0.5))
        # filter shape : [batch, heads, q_len, k_len]

        att = torch.einsum('bhqk,bhvd->bhqd',[filter,V])
        att  = att.reshape(query_len, batch, self.embedding_size)

        # att shape : [query_len, batch, embedding_size]

        return self.fc(att)

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, p, forward_expansion=4):
        super(EncoderBlock, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(embedding_size, heads)
        self.Norm1 = nn.LayerNorm(embedding_size)
        self.Norm2 = nn.LayerNorm(embedding_size)
        self.Dropout = nn.Dropout(p)
        self.FeedForward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion*embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embedding_size, embedding_size)
        )
    
    def forward(self, x, mask):
        # x shape : [seq_len, batch, embedding_size]
        att = self.Dropout(self.MultiHeadAttention(x,x,x,mask))
        norm1 = self.Norm1(torch.add(x,att))
        # norm1 shape : [query_len, batch, embedding_size]

        ff = self.Dropout(self.FeedForward(norm1))
        norm2 = self.Norm2(torch.add(norm1,ff))
        # norm2 shape : [query_len, batch, embedding_size]

        return norm2

class Encoder(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_size, 
                 heads, 
                 layers, 
                 p, 
                 max_length
                 ):
        super(Encoder, self).__init__()
        self.d = embedding_size ** 0.5
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = PositionalEncoder(embedding_size)

        self.layers = nn.ModuleList([EncoderBlock(embedding_size, heads, p) for _ in range(layers)])
        self.dropout = nn.Dropout(p)


    def forward(self, x, mask):
        # x shape : [seq_len, batch]

        x = self.embeddings(x) # * self.d # to avoid values being to small compared to the positional encoding
        # x shape : [seq_len, batch, embedding_size]

        x = self.dropout(self.position_embeddings(x))
        # x shape : [seq_len, batch, embedding_size]

        for layer in self.layers:
            x = layer(x, mask)
        # x shape : [seq_len, batch, embedding_size]
        
        return x


class MaskedMultiHeadBlock(nn.Module):
    def __init__(self, embedding_size, heads, p):
        super(MaskedMultiHeadBlock, self).__init__()
        self.MaskedMultiHeadAttention = MultiHeadAttention(embedding_size,heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask):
        # q,k,v shape : [q_len=k_len=v_len, batch, embedding_size]

        x = self.MaskedMultiHeadAttention(q, k, v, mask)
        norm = self.norm(self.dropout(torch.add(q,x)))
        # norm shape : [q_len, batch, embedding_size]

        return norm

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, p, forward_expansion=4):
        super(DecoderBlock, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(embedding_size, heads)
        self.Norm1 = nn.LayerNorm(embedding_size)
        self.Norm2 = nn.LayerNorm(embedding_size)
        self.Dropout = nn.Dropout(p)
        self.FeedForward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion*embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embedding_size, embedding_size)
        )
        self.MaskedAttention = MaskedMultiHeadBlock(embedding_size, heads, p)
    
    def forward(self, x, k, v, mask):
        # x shape : [seq_len, batch, embedding_size]
        # q, k -> from Encoder
        q = self.MaskedAttention(x, x, x, mask)

        att = self.Dropout(self.MultiHeadAttention(q, k, v))
        norm1 = self.Norm1(torch.add(q,att))
        # norm1 shape : [query_len, batch, embedding_size]

        ff = self.Dropout(self.FeedForward(norm1))
        norm2 = self.Norm2(torch.add(norm1,ff))
        # norm2 shape : [query_len, batch, embedding_size]

        return norm2
        
class Decoder(nn.Module):
        def __init__(self, 
                     vocab_target_size, 
                     embedding_size, 
                     heads, 
                     layers, 
                     p, 
                     max_length
        ):
            super(Decoder, self).__init__()
            self.embeddings = nn.Embedding(vocab_target_size, embedding_size)
            self.position_embeddings = PositionalEncoder(embedding_size)

            self.layers = nn.ModuleList([DecoderBlock(embedding_size, heads, p) for _ in range(layers)])

            self.Linear = nn.Linear(embedding_size, vocab_target_size)
            self.Softmax = nn.Softmax(dim=2)

        def forward(self, x, k, v, mask):
            # x shape : [seq_len, batch]
            x = self.embeddings(x)
            x = self.position_embeddings(x)
            # x shape : [seq_len, batch, embedding_size]

            for layer in self.layers:
                dec = layer(x, k, v, mask)
            # dec shape : [v_len, batch, embedding_size]

            out = self.Softmax(self.Linear(dec))
            # out shape : [1, batch , vocab_target_size] 

            return out


class Transformer(nn.Module):
    def __init__(self, 
                 trg_vocab_size,
                 src_vocab_size,
                 embedding_size,
                 heads,
                 layers,
                 p,
                 max_length,
                 pad_idx,
                 device
                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, 
                               embedding_size, 
                               heads, 
                               layers, 
                               p, 
                               max_length
                               )
        
        self.decoder = Decoder( trg_vocab_size, 
                                embedding_size, 
                                heads, 
                                layers, 
                                p, 
                                max_length
                                )
        
        self.pad_idx = pad_idx
        self.vocab_target_size = trg_vocab_size
        self.device = device

    def forward(self, source, target, force_ratio=0.5):
        # source shape : [src_seq_len, batch]
        src_mask = self.padding_mask(source)
        enc_out = self.encoder(source, src_mask)
        k = v = enc_out

        trg_len, batch = target.shape
        outputs = torch.zeros([trg_len, batch, self.vocab_target_size])

        x = target[0].unsqueeze(0) # first word "<sos>" shape : [1,batch]
        
        for step in range(1,trg_len):
            mask = self.make_trg_mask(x)
            dec_out = self.decoder(x, k, v, mask)
            dec_out = dec_out[-1].unsqueeze(0)
            outputs[step] = dec_out
            
            if random.random() > force_ratio:
                x = torch.cat([x,dec_out.argmax(2)], dim=0)
            else:
                x = torch.cat([x,target[step].unsqueeze(0)], dim=0)

        return x, outputs


    def padding_mask(self, src):
        src_mask = (src != self.pad_idx).permute(1,0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_len, N = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.long()
      
if __name__ == "__main__":
    ###### TEST #######
    src_vocab_size = 50
    trg_vocab_size = 64
    embedding_size = 20
    heads = 2
    layers = 2
    batch = 4
    p = 0.1
    max_length = 40
    pad_idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    src = torch.rand([33,batch]).long()
    trg = torch.rand([25,batch]).long()
    e = Encoder(src_vocab_size, 
                embedding_size, 
                heads, 
                layers, 
                p, 
                max_length
                )


    T = Transformer(src_vocab_size, 
                    trg_vocab_size,
                    embedding_size,
                    heads,
                    layers,
                    p,
                    max_length,
                    pad_idx,
                    device
                    ).to(device)


    outs, loss = T(src,trg)
    print(outs.shape)
    print(loss.shape)
