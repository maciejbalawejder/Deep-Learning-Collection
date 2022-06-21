import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

class TransformerConfig:
    d : int = 512
    heads : int = 8
    dff : int = 2048
    N : int = 3
    p : float = 0.1
    device : str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embedding(nn.Module):
    " Embedding layer with scalling and dropout. "
    def __init__(
        self,
        d : int,
        vocab_size : int,
        ):
        super().__init__()
        self.d = d
        self.embedding = nn.Embedding(vocab_size, d)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d)

class PE(nn.Module):
    "The Positional Encoding Module with dropout layer. "
    def __init__(
        self,
        d : int,
        p : float,
        max_len = 5_000
        ):    
        super().__init__()

        self.pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        div = torch.exp(- torch.arange(0, d, 2)* math.log(10000) / d)
        self.pe[:, 0::2] = torch.sin(pos / div)
        self.pe[:, 1::2] = torch.cos(pos / div)

        self.pe = self.pe.unsqueeze(1)
        self.dropout = nn.Dropout(p)
        self.register_buffer('pos_embedding', self.pe)

    def forward(self, x: Tensor) -> Tensor:
        " x: shape : [seq_len, batch] "
        return self.dropout(x + self.pe[:x.shape[0], :, :])

class SelfAttention(nn.Module):
    " Multi head self-attention sub-layer followed by Add&Norm layer. "
    def __init__(
        self, 
        heads : int,
        d : int,
        p : float
        ):

        super().__init__()

        self.heads = heads
        self.head_dim = d // heads
        self.d = d

        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.linear = nn.Linear(self.d, self.d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        batch = q.shape[1]
        q_len = q.shape[0]
        k_len = k.shape[0]
        v_len = v.shape[0]
        
        Q = self.Q(q.reshape(batch, q_len, self.heads, self.head_dim))
        K = self.K(k.reshape(batch, k_len, self.heads, self.head_dim))
        V = self.V(v.reshape(batch, v_len, self.heads, self.head_dim))

        QK = torch.einsum("bqhd, bkhd -> bhqk", [Q, K])
       

        if mask is not None:
            QK = QK.masked_fill(mask == 0, float("-inf"))
        
        scale = QK / math.sqrt(self.d)
        softmax = self.dropout(F.softmax(scale, dim=-1))
        output = torch.einsum("bhqk, bvhd -> bqhd", [softmax, V])
        concat = output.reshape(q_len, batch, self.d)
        return self.linear(concat)
        
class FeedForward(nn.Module):
    " Position-wise fully conntected feed-forward network with 2 linear transformations, where first is followed by GELU(inspiration from GPT) activation with Add&Norm operation."
    def __init__(
        self,
        d : int,
        dff : int,
        p : float
        ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d, dff),
            nn.GELU(),
            nn.Linear(dff, d),
            nn.Dropout(p)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)

class EncoderLayer(nn.Module):
    "Encoder layer with two sub-layers multi-head attention and position-wise fully conntected feed-forward network. "
    def __init__(
        self,
        heads : int,
        d : int,
        dff : int,
        p : float
        ):
        super().__init__()

        self.attention = SelfAttention(heads, d, p)
        self.ff = FeedForward(d, dff, p)
        self.l1 = nn.LayerNorm(d)
        self.l2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        att_out = self.attention(x, x, x, src_mask)
        addnorm1 = self.l1(x + self.dropout(att_out))
        ff_out = self.ff(addnorm1)
        addnorm2 = self.l2(addnorm1 + self.dropout(ff_out))
        return addnorm2

class Encoder(nn.Module):
    " Encoder with N-Encoding layers. "
    def __init__(
        self,
        N : int,
        heads : int,
        d : int,
        dff : int,
        p : float
        ):
        super().__init__()

        self.encoder = nn.ModuleList([EncoderLayer(heads, d, dff, p) for _ in range(N)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        for enc_layer in self.encoder:
            x = enc_layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    "Decoder layer with three sub-layers, two multi-head attention module and position-wise fully conntected feed-forward network on the top."
    def __init__(
        self,
        heads : int,
        d : int,
        dff : int,        
        p : float
        ):
        super().__init__()

        self.masked_attention = SelfAttention(heads, d, p)
        self.enc_dec_attention = SelfAttention(heads, d, p)
        self.ff = FeedForward(d, dff, p)
        
        self.l1 = nn.LayerNorm(d)
        self.l2 = nn.LayerNorm(d)
        self.l3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)

    def forward(self, x: Tensor, k: Tensor, v: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tensor:
        q = self.masked_attention(x, x, x, trg_mask)
        addnorm1 = self.l1(self.dropout(q) + x)
        enc_dec_out = self.enc_dec_attention(addnorm1, k, v, src_mask)
        addnorm2 = self.l2(self.dropout(enc_dec_out) + addnorm1)
        ff_out = self.ff(addnorm2)
        addnorm3 = self.l3(self.dropout(ff_out) + addnorm2)
        return addnorm3

class Decoder(nn.Module):
    " Decoder with N-Encoding layers. "
    def __init__(
        self,
        N : int,
        heads : int,
        d : int,
        dff : int,
        p : float
        ):
        super().__init__()

        self.decoder = nn.ModuleList([DecoderLayer(heads, d, dff, p) for _ in range(N)])

    def forward(self, x: Tensor, k: Tensor, v: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tensor:
        for dec_layer in self.decoder:
            x = dec_layer(x, k, v, trg_mask, src_mask)
        return x

            
class Transformer(nn.Module):
    def __init__(
        self,
        config,
        src_vocab_size : int,
        trg_vocab_size : int,
        src_pad : int,
        trg_pad : int
        ):
        super().__init__()

        self.src_emb = Embedding(config.d, src_vocab_size)
        self.trg_emb = Embedding(config.d, trg_vocab_size)
        self.pos_emb = PE(config.d, config.p)
        self.encoder = Encoder(config.N, config.heads, config.d, config.dff, config.p)
        self.decoder = Decoder(config.N, config.heads, config.d, config.dff, config.p)
        self.head = nn.Linear(config.d, trg_vocab_size, bias=False)

        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.config = config

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        src_mask = self.get_src_mask(src, self.src_pad)
        trg_mask = self.get_trg_mask(trg)
        src = self.pos_emb(self.src_emb(src))
        trg = self.pos_emb(self.trg_emb(trg))
        # src and trg shape: [seq_len, batch, d(embedding_size)]

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, enc_out, trg_mask, src_mask)
        # dec_out shape: [seq_len, batch, d]
        return self.head(dec_out)
    
    def encode(self, src: Tensor):
        src_mask = self.get_src_mask(src, self.src_pad)
        return self.encoder(self.pos_emb(self.src_emb(src)), src_mask)

    def decode(self, trg: Tensor, enc_out: Tensor, src_mask: Tensor):
        trg_mask = self.get_trg_mask(trg, self.trg_pad)
        return self.decoder(self.pos_emb(self.trg_emb(trg)), enc_out, enc_out, trg_mask, src_mask)

    def get_src_mask(self, src, pad_idx):
        # src shape: [seq_len, batch]
        seq_len, batch = src.shape
        src_mask = src != pad_idx
        return src_mask.reshape(batch, 1, 1, seq_len).to(self.config.device)
    
    def get_trg_mask(self, trg):
        seq_len, batch = trg.shape
        trg_mask = torch.triu(torch.zeros((seq_len, seq_len))==1).expand(batch, 1, seq_len, seq_len).to(self.config.device)
        return trg_mask
        



if __name__ == "__main__":

    src = torch.randint(0, 100, (4, 1))
    trg = torch.randint(0, 50, (2, 1))
    t = Transformer(
        TransformerConfig,
        src_vocab_size = 100,
        trg_vocab_size = 50,
        src_pad = 1,
        trg_pad = 1
        )

    print(t(src, trg).shape)
    print(torch.backends.mps.is_available())
    print(torch.__version__)
