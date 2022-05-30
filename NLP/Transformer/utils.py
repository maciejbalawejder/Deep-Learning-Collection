import torch
from torch import Tensor
import torch.nn.functional as F

class GreedyDecoder:
    """
    Greedy Decoder is an simple auto-regressive function to generated translated sentances.
    First we get the contextualize output from the encoder.
    Then we sequentially feed the transformer decoder with newly generated words with highest probability score.  
    """
    def __init__(
            self,
            first_token : int,
            end_token : int,
            max_len : int = 50
            ):
        self.max_len = max_len
        self.translation = torch.LongTensor([first_token]).unsqueeze(0)
        self.end_token = torch.tensor(end_token).unsqueeze(0)
    
    def __call__(self, model: torch.nn.Module, src: Tensor) -> Tensor:
        dec_out = model.decoder(self.translation, src, None)
        output = model.classifier(dec_out)
        logits = F.softmax(output, dim=-1)
        words = torch.argmax(logits, dim=-1)
        last_word = words[:, -1].unsqueeze(0)
        self.translation = torch.concat([self.translation, last_word], dim=-1)

        for _ in range(self.max_len):
            dec_out = model.decoder(self.translation, src, None)
            output = model.classifier(dec_out)
            logits = F.softmax(output, dim=-1)
            words = torch.argmax(logits, dim=-1)
            last_word = words[:, -1].unsqueeze(0)
            self.translation = torch.concat([self.translation, last_word], dim=-1)

            if last_word == self.end_token:
                break            