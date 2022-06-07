import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(
        self,
        config
        ):

        super().__init__()

        self.C = nn.Linear(config.d_model, config.d_model*3)
        self.linear = nn.Linear(config.d_model, config.d_model)

        self.FF = nn.Sequential(
            nn.Linear(config.d_model, config.inner_state),
            nn.GELU(),
            nn.Linear(config.inner_state, config.d_model),
            nn.Dropout(config.p)
        )

        self.LN1 = nn.LayerNorm(config.d_model)
        self.LN2 = nn.LayerNorm(config.d_model)

        self.head_dim = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = nn.Dropout(config.p)

    def forward(self, x: Tensor) -> Tensor:
        batch, window, d = x.shape
        mask = self._make_mask(batch, window)
        
        c = self.C(x)
        q, k, v = torch.split(c, d, 2)
        q = q.reshape(batch, window, self.heads, self.head_dim)
        k = k.reshape(batch, window, self.heads, self.head_dim)
        v = v.reshape(batch, window, self.heads, self.head_dim)

        QK = torch.einsum("bqhd, bkhd -> bhqk", [q, k]) / math.sqrt(d)
        QK = QK.masked_fill(mask==0, float("-inf"))
        scores = self.dropout(F.softmax(QK, dim=3))
        output = torch.einsum("bhqk, bvhd -> bqhd", [scores, v])
        concat = output.reshape(batch, window, d)
        linear = self.dropout(self.linear(concat))

        addnorm1 = self.LN1(x + linear)
        addnorm2 = self.LN2(addnorm1 + self.FF(addnorm1))
        return addnorm2

    def _make_mask(self, batch : Tensor, window : int):
        mask = torch.tril(torch.ones((window, window)))
        return mask.reshape(batch, 1, window, window)

class GPT(nn.Module):
    def __init__(
        self,
        config
        ):

        super().__init__()
        
        self.word_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.window, config.d_model)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])
        self.classifier = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.p)
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        batch, window = x.shape
        positions = torch.arange(0, window).expand(batch, window).to(self.config.device) 
        emb = self.dropout(self.word_emb(x) + self.pos_emb(positions))

        for layer in self.decoder:
            emb = layer(emb)

        output = self.classifier(emb)
        return output

from config import Config
if __name__ == "__main__":
    config = Config()
    print(config)
    gpt = GPT(config)
    gpt(torch.randint(0, config.vocab_size, (1, config.window))).shape