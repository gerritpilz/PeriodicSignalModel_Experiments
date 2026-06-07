import torch
from torch.nn import functional as F
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from times_model import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
n_channels      = 38
seq_len         = 100
pred_len        = 50
d_embd          = 64
dropout         = 0.1
n_timeBlocks    = 3
k_periods       = 3
sigma           = 0.5
alpha           = 0.5
lr              = 1e-3
lr_min          = 1e-5
scheduler_steps = 100
batch_size      = 32
n_epochs        = 100
eval_iter       = 10
# -----------------------

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

def train(train_path, val_path):
    train_file = pd.read_csv(train_path, header=None)
    val_file   = pd.read_csv(val_path,   header=None)

    train_data = torch.tensor(train_file.values, dtype=torch.float32)
    val_data   = torch.tensor(val_file.values,   dtype=torch.float32)

    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset   = TimeSeriesDataset(val_data,   seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    net = model(
        n_channels,
        seq_len,
        d_embd,
        dropout,
        n_timeBlocks,
        k_periods,
        sigma,
        alpha
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_steps,
        eta_min=lr_min
    )

    @torch.no_grad()
    def estimate_loss():
        net.eval()
        losses = {'train': [], 'val': []}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb)[:, -pred_len:, :]
                losses[split].append(F.mse_loss(pred, yb).item())
        net.train()
        return {
            'train': sum(losses['train']) / len(losses['train']),
            'val':   sum(losses['val']) / len(losses['val'])
        }

    for epoch in range(n_epochs):
        net.train()
        for it, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)[:, -pred_len:, :]
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % eval_iter == 0:
                losses = estimate_loss()
                print(f"epoch {epoch} step {it}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to training CSV')
    parser.add_argument('--val', required=True, help='Path to validation CSV')
    args = parser.parse_args()

    train(args.train, args.val)