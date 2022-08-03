import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn.functional as F
from config import Config

class DecoderLayer(nn.Module):
    def __init__(
        self,
        config
        ):

        super().__init__()

        " Masked Multi Self Attention."
        self.C = nn.Linear(config.d_model, config.d_model*3)
        self.linear = nn.Linear(config.d_model, config.d_model)

        " Feed Forward Module."
        self.FF = nn.Sequential(
            nn.Linear(config.d_model, config.inner_state),
            nn.GELU(),
            nn.Linear(config.inner_state, config.d_model),
            nn.Dropout(config.p)
        )

        " Two Layer Norms."
        self.LN1 = nn.LayerNorm(config.d_model)
        self.LN2 = nn.LayerNorm(config.d_model)

        self.head_dim = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = nn.Dropout(config.p)

        " Weight Initialization N[0, 0.02] "
        nn.init.normal_(self.FF[0].weight, 0, 0.02)
        nn.init.normal_(self.FF[2].weight, 0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, window, d = x.shape
        mask = self._make_mask(batch, window)
        
        c = self.C(x)
        q, k, v = torch.split(tensor=c, split_size_or_sections=d, dim=2)
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

    def _make_mask(self, batch, window):
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
        self.dropout = nn.Dropout(config.p)
        self.config = config

        nn.init.normal_(self.word_emb.weight, 0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, window = x.shape
        positions = torch.arange(0, window).expand(batch, window).to(self.config.device) 
        dec_out = self.dropout(self.word_emb(x) + self.pos_emb(positions))

        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out)

        return dec_out

class LMHead(nn.Module):
    def __init__(
        self,
        config,
        gpt
        ):

        super().__init__()
        self.gpt = gpt
        self.prediction = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.prediction.weights = gpt.word_emb.weight

    def forward(self, x: Tensor) -> Tensor:
        dec_out = self.gpt(x)
        logits = self.prediction(dec_out)
        return logits

class CLSHead(nn.Module):
    def __init__(
        self,
        config,
        gpt
        ):

        super().__init__()
        self.gpt = gpt
        self.prediction = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.prediction.weights = gpt.word_emb.weight
        self.classifier = nn.Linear(config.d_model, config.n_class)

        nn.init.normal_(self.classifier.weight, std=0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        dec_out = self.gpt(x)

        lm_logits = self.prediction(dec_out)
        cls_logits = self.classifier(dec_out)
        return lm_logits, cls_logits

if __name__ == "__main__":
    config = Config()
    gpt = GPT(config)
    lm_test = LMHead(config, gpt)
    cls_test = CLSHead(config, gpt)
    logits = lm_test(torch.randint(0, config.vocab_size, (1, config.window)))
    print(logits.shape)
    lm_logits, cls_logits = cls_test(torch.randint(0, config.vocab_size, (1, config.window)))
    print(lm_logits.shape, cls_logits.shape)
