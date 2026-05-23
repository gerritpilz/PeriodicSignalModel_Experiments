import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training
n_epochs = 4
eval_iter = 5

# hyperparameters
n_channels = 10  # channels from dataset
seq_len = 128
pred_len = 32
batch_size = 16
d_embd = 128
dropout = 0.2

# model
n_timeBlocks = 8
k_periods = 6

# gaussian bandpass filter
sigma = 0.5


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.X = data
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
        losses = torch.zeros(eval_iter, device=device)
        for it, (xb, yb) in enumerate(loader):
            if it == eval_iter:
                break

            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            pred = pred[:, -pred_len:, :]

            loss = F.mse_loss(pred, yb)
            losses[it] = loss
        out[split] = losses.mean()
    model.train()
    return out

# read files
train_file = pd.read_csv('machine-1-1_train.txt', header=None, nrows=20000)
val_file = pd.read_csv('machine-1-1_val.txt', header=None, nrows=20000)

train_data = torch.tensor(train_file.values, dtype=torch.float32)
val_data = torch.tensor(val_file.values, dtype=torch.float32)

print(train_data.std())


# Dataset, Dataloader
train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# create model
model = model(n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, sigma)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=2e-5)

# training loop
model.train()
for epoch in range(n_epochs):
    for it, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        pred = pred[:, -pred_len:, :]
        loss = F.mse_loss(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % eval_iter == 0:
            losses = estimate_loss()
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if it % 20 == 0:
                x_sample = xb[0:1]  # one sample

                # here: take first frequency
                periods, freq_bins, _ = model.blocks[0].get_periods(x_sample)

                amps = model.blocks[0].compute_band_amplitude(x_sample, freq_bins[0])

                plt.figure(figsize=(8, 4))
                plt.plot(x_sample[0, :, 0].detach().cpu(), label="original")
                plt.plot(amps[0, :, 0].detach().cpu(), label="filtered")
                plt.legend()
                plt.title(f"Step {it}")

                plt.savefig(f"/content/plot_{it}.png")
                plt.close()

# generate from model
model.eval()
context = data[-seq_len:, :]


