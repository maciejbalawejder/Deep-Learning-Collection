from utils import load_checkpoint, save_checkpoint, i2w, datasetGenerator
from seq2seqattention import Seq2Seq, Encoder, Decoder
from model import Model
import torch

BATCH = 32
LEARNING_RATE = 2e-4
LAYERS_ENC = 1
HIDDEN_DIM = 512
EPOCHS = 10
EMBEDDING_DIM = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 0.5

# DATASET
(train_iterator, validation_iterator, test_iterator), (train_data, validation_data, test_data), (english, german) = datasetGenerator(BATCH, DEVICE, 10_000)

# MODEL
ENCODER = Encoder(input_size=len(german.vocab), embedding_size=EMBEDDING_DIM, 
                  hidden_size=HIDDEN_DIM, layers=LAYERS_ENC, p=P).to(DEVICE)

DECODER = Decoder(input_size=len(english.vocab), embedding_size=EMBEDDING_DIM, 
                  hidden_size=HIDDEN_DIM, layers=1, p=0.1).to(DEVICE)

SEQ2SEQ = Seq2Seq(ENCODER, DECODER).to(DEVICE)
MODEL = Model(model=SEQ2SEQ, lr=LEARNING_RATE, vocab=english, device=DEVICE)
lambda2 = lambda epoch: 0.5 ** epoch
STEPLR = torch.optim.lr_scheduler.LambdaLR(MODEL.opt, lr_lambda=lambda2)

# TRAINING LOOP
force_ratio = 0.5
train_loss, train_bleu = [], []
val_loss, val_bleu = [0], [] # added zero at the start to avoid error with max
lr = []
for epoch in tqdm_notebook(range(1, 30+1), desc = 'Epoch'):
    tl, tb = MODEL.train_step(train_iterator, force_ratio)
    vl, vb = MODEL.validation_step(validation_iterator)

    if vl < min(val_loss):
        save_checkpoint(MODEL.checkpoint, 'drive/MyDrive/attentionv1')

    if vl > val_loss[-1] and epoch > 1:
        print("Overfitting...")
        break
    
    if epoch%4==0:
        STEPLR.step()

    train_loss.append(tl)
    train_bleu.append(tb)
    val_loss.append(vl)
    val_bleu.append(vb)
    lr.append(STEPLR.get_lr())
    
    print(f'Loss: {tl}, {vl}')
    print(f'Bleu score: {tb}, {vb}')
    print(f'Learning rate: {lr[-1]} Force ratio : {force_ratio}')

print(MODEL.test_step(test_iterator))
