import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from times_model import times_model
from benchmark_model import model
from config import base_config
from types import SimpleNamespace
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(net, epoch, config):
    path = f'/content/drive/MyDrive/checkpoints/times_model_{epoch}.pt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': net.state_dict(),
        'model_config': {
            'n_channels': config.n_channels,
            'seq_len': config.seq_len,
            'pred_len': config.pred_len,
            'd_embd': config.d_embd,
            'dropout':  config.dropout,
            'n_timeBlocks': config.n_timeBlocks,
            'k_periods': config.k_periods,
            'sigma': config.sigma
        }
    }, path)

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

def train(train_path, val_path, config):
    # Data
    train_file = pd.read_csv(train_path, header=None, nrows=20000)
    val_file   = pd.read_csv(val_path,   header=None, nrows=20000)

    train_data = torch.tensor(train_file.values, dtype=torch.float32)
    val_data   = torch.tensor(val_file.values, dtype=torch.float32)

    # Dataset
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len)
    val_dataset   = TimeSeriesDataset(val_data,   config.seq_len, config.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size)

    # Model
    net = times_model(
        config.n_channels,
        config.seq_len,
        config.d_embd,
        config.dropout,
        config.n_timeBlocks,
        config.k_periods,
        config.sigma,
        config.alpha
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr)

    total_steps = config.n_epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.lr_min
    )

    # Evaluation
    @torch.no_grad()

    def estimate_loss():
        net.eval()
        losses = {'train': [], 'val': []}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            for i, (xb, yb) in enumerate(loader):

                if i > 4:
                    break

                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb)[:, -config.pred_len:, :]
                losses[split].append(F.mse_loss(pred, yb).item())

        net.train()

        return {
            'train': sum(losses['train']) / len(losses['train']),
            'val': sum(losses['val']) / len(losses['val'])
        }

    # Training
    for epoch in range(config.n_epochs):
        save_model(net, epoch, config)
        net.train()

        for it, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            pred = net(xb)
            pred = pred[:, -config.pred_len:, :]

            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % config.eval_iter == 0:
                l = estimate_loss()
                print(f"epoch {epoch} step {it}: train loss {l['train']:.7f} | val loss {l['val']:.7f}")

        save_model(net, epoch, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to training CSV')
    parser.add_argument('--val', required=True, help='Path to validation CSV')
    args = parser.parse_args()

    train(args.train, args.val, SimpleNamespace(**base_config))







