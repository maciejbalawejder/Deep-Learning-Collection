import torch
import torch.nn as nn
import numpy as np
from torchtext.data.metrics import bleu_score

class Model:
    def __init__(self, model, lr, vocab, device):
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.vocab.stoi["<pad>"]) # ignore paddings while calculating loss
        self.vocab = vocab
        self.device = device

    def train_step(self, dataset, force_ratio):
        epoch_loss = []
        bleu = []
        self.model.train()
        for batch in dataset:
            src = batch.src.to(self.device)
            trg = batch.trg.to(self.device)
            self.opt.zero_grad()
            outputs = self.model(src, trg, force_ratio=force_ratio).to(self.device)
            loss = self.loss(outputs[1:].reshape(-1,outputs.shape[2]), trg[1:].reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1) # clipping gradients
            self.opt.step()
            epoch_loss.append(loss.item())

        self.checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.opt.state_dict()}

        return np.mean(epoch_loss), np.mean(bleu)*100

    def validation_step(self, dataset):
        bleu = []
        epoch_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                src = batch.src.to(self.device)
                trg = batch.trg.to(self.device)
                outputs = self.model(src, trg, force_ratio=0).to(self.device)
                loss = self.loss(outputs[1:].reshape(-1,outputs.shape[2]), trg[1:].reshape(-1))
                epoch_loss.append(loss.item())

                source = i2w(trg, self.vocab)
                target = i2w(outputs.argmax(2), self.vocab, target=True)
            
                bleu.append(bleu_score(source, target))


        return np.mean(epoch_loss), np.mean(bleu) * 100

    def test_step(self, dataset):
        bleu = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                src = batch.src.to(self.device)
                trg = batch.trg.to(self.device)
                outputs = self.model(src, trg, force_ratio=0).to(self.device)

                source = i2w(trg, self.vocab)
                target = i2w(outputs.argmax(2), self.vocab, target=True)
                score = bleu_score(source, target)
                bleu.append(score)
        return np.mean(bleu) * 100