import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import model

# hyperparameters
n_channels = 10
seq_len = 128
pred_len = 32
batch_size = 16
d_embd = 64
dropout = 0.2

# model
n_timeBlocks = 12
k_periods = 8

# filter
bw = 1

# Attention
n_heads = 4

# training
n_epochs = 4
eval_iter = 5


class weatherDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.X[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iter, device='cuda')
        for it, (xb, yb) in enumerate(loader):
            if it == eval_iter:
                break

            xb, yb = xb.to('cuda'), yb.to('cuda')
            pred = model(xb)
            pred = pred[:, -pred_len:, :]

            loss = F.mse_loss(pred, yb)
            losses[it] = loss
        out[split] = losses.mean()
    model.train()
    return out

# read file
file = pd.read_csv('weather_prediction_dataset.csv', usecols=range(104, 114)) #  munich weather data

# Train and test splits
data = torch.tensor(file.values, dtype=torch.float32)  # (T, C)
n = int(0.9*data.shape[0])
train_data = data[:n]
val_data = data[n:]

# Normalize data
mean = train_data.mean(dim=0)
std = train_data.std(dim=0)
train_data = (train_data-mean) / std
val_data = (val_data-mean) / std

# Dataset, Dataloader
train_dataset = weatherDataset(train_data, seq_len, pred_len)
val_dataset = weatherDataset(val_data, seq_len, pred_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# create model
model = model(n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, bw, n_heads)
model = model.to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# training loop
model.train()
for epoch in range(n_epochs):
    for it, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to('cuda'), yb.to('cuda')
        pred = model(xb)
        pred = pred[:, -pred_len:, :]
        loss = F.mse_loss(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_iter == 0:
            losses = estimate_loss()
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# generate from model
model.eval()
context = data[-seq_len:, :]


