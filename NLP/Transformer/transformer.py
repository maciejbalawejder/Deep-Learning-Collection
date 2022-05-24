import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

heads = 8
d = 512 # embedding dim
dff = 2048 # expansion dim
N = 6 # layers
p = 0.1 # dropout rate

src = torch.randint(0, 100, (1, 4))
trg = torch.randint(0, 50, (1, 2))

class Embedding(nn.Module):
    " Embedding layer with scalling and dropout. "
    def __init__(
        self,
        d : int,
        vocab_size : int,
        ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

e = Embedding(d, 100)
e(src).shape

class PE(nn.Module):
    "Implement the Positional Encoding function with dropout. "
    def __init__(
        self,
        d : int,
        p : int,
        max_len = 100
        ):    
        super().__init__()

        self.pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        div = torch.pow(10_000, 2 * torch.arange(0, d, 2) / d)
        self.pe[:, 0::2] = torch.sin(pos / div)
        self.pe[:, 1::2] = torch.cos(pos / div)

        self.dropout = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:x.shape[1]])

pe = PE(d, p)
pe(e(src)).shape

class SelfAttention(nn.Module):
    " Multi head self-attention sub-layer followed by Add&Norm layer. "
    def __init__(
        self, 
        heads : int,
        d : int,
        p : int = 0.1
        ):

        super().__init__()

        self.heads = heads
        self.head_dim = d // heads
        self.d = d

        self.Q = nn.Linear(self.head_dim, self.head_dim)
        self.K = nn.Linear(self.head_dim, self.head_dim)
        self.V = nn.Linear(self.head_dim, self.head_dim)

        self.linear = nn.Linear(self.d, self.d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        batch = q.shape[0]
        q_len = q.shape[1]
        k_len = k.shape[1]
        v_len = v.shape[1]
        
        Q = self.Q(q.reshape(batch, q_len, self.heads, self.head_dim))
        K = self.K(k.reshape(batch, k_len, self.heads, self.head_dim))
        V = self.V(v.reshape(batch, v_len, self.heads, self.head_dim))

        QK = torch.einsum("bqhd, bkhd -> bhqk", [Q, K])
        scale = QK / math.sqrt(self.d)

        if mask is not None:
            scale = scale.masked_fill(mask == 0, float("-inf"))

        softmax = F.softmax(scale, dim=3)
        output = torch.einsum("bhqk, bvhd -> bqhd", [softmax, V])
        concat = output.reshape(batch, q_len, self.d)
        linear = self.linear(concat)
        addnorm = q + self.dropout(self.norm(linear))

        return addnorm

s = SelfAttention(heads, d)
x = pe(e(src))
s(x, x, x).shape

class FeedForward(nn.Module):
    " Position-wise fully conntected feed-forward network with 2 linear transformations, where first is followed by ReLU activation with Add&Norm operation."
    def __init__(
        self,
        d : int,
        dff : int,
        p : int = 0.1
        ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d, dff),
            nn.ReLU(),
            nn.Linear(dff, d)
        )

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x: Tensor) -> Tensor:
        " Applying norm before the *add* operation empirically yields better results. "
        norm = self.norm(self.ff(x))
        return x + self.dropout(norm)

f = FeedForward(d, dff)
x = s(x, x, x)
f(x).shape

class EncoderLayer(nn.Module):
    "Encoder layer with two sub-layers multi-head attention and position-wise fully conntected feed-forward network. "
    def __init__(
        self,
        heads : int,
        d : int,
        dff : int
        ):
        super().__init__()

        self.attention = SelfAttention(heads, d)
        self.ff = FeedForward(d, dff)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.ff(self.attention(q, k, v))

enc = EncoderLayer(heads, d, dff)
x = pe(e(src))
enc(x, x, x).shape

class DecoderLayer(nn.Module):
    "Decoder layer with three sub-layers, two multi-head attention mechanisms and position-wise fully conntected feed-forward network on the top."
    def __init__(
        self,
        heads : int,
        d : int,
        dff : int        
        ):
        super().__init__()

        self.masked_attention = SelfAttention(heads, d)
        self.enc_layer = EncoderLayer(heads, d, dff)

    def forward(self, x: Tensor, k: Tensor, v: Tensor, trg_mask: Tensor) -> Tensor:
        q = self.masked_attention(x, x, x, trg_mask)
        return self.enc_layer(q, k, v)

class EncoderDecoder(nn.Module):
    " Encoder-Decoder archiecture without Positional Encoding nor Embeddings."
    def __init__(
        self,
        heads : int,
        d : int,
        dff : int,
        N : int,
        src_pad_idx : int,
        trg_pad_idx : int
        ):
        super().__init__()

        self.enc_layer = nn.ModuleList([EncoderLayer(heads, d, dff) for _ in range(N)])
        self.dec_layer = nn.ModuleList([DecoderLayer(heads, d, dff) for _ in range(N)])
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        for enc in self.enc_layer:
            src = enc(src, src, src, self.src_pad_idx)

        for dec in self.dec_layer:
            trg = dec(trg, src, src, self._make_mask(trg, self.trg_pad_idx))
        
        return trg

    def _make_pad_mask(self, x, pad_idx):
        batch, seq_len = x.shape
        mask = x != pad_idx
        return mask.reshape(batch, 1, 1, seq_len)

    def _make_trg_mask(self, trg, pad_idx):
        # trg shape : [1, 2, 512]
        pad_mask = self._make_pad_mask(trg, pad_idx)
        batch, trg_len, _ = trg.shape
        mask = torch.tril(torch.ones(trg_len, trg_len))
        mask = mask.reshape(batch, 1, trg_len, trg_len)
        return mask & pad_mask

class Classifier(nn.Module):
    " The last stage of transformer architecure where the output of decoder is passed through linear layer."
    def __init__(
        self,
        d : int,
        trg_vocab_size : int
        ):
        super().__init__()
        
        self.linear = nn.Linear(d, trg_vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(
        self,
        d : int,
        heads : int, 
        dff : int,
        N : int,
        src_vocab_size : int,
        trg_vocab_size : int,
        p : int,
        src_pad : int,
        trg_pad : int
        ):
        super().__init__()

        self.encdec = EncoderDecoder(heads, d, dff, N, src_pad, trg_pad)
        self.pe = PE(d, p)
        self.src_embeddings = Embedding(d, src_vocab_size)
        self.trg_embeddings = Embedding(d, trg_vocab_size)
        self.classifier = Classifier(d, trg_vocab_size)

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        src = self.pe(self.src_embeddings(src))
        trg = self.pe(self.trg_embeddings(trg))
        output = self.encdec(src, trg)
        return self.classifier(output)

t = Transformer(
    d = d,
    heads = heads,
    dff = dff,
    N = N,
    src_vocab_size = 100,
    trg_vocab_size = 50,
    p = p,
    src_pad = 1,
    trg_pad = 1
    )

t(src, trg).shape
