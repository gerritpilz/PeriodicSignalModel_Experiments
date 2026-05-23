import torch
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import wandb
from model import model
from config import base_config, sweep_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(config):
    # Data
    train_file = pd.read_csv('machine-1-1_train.txt', header=None, nrows=20000)
    val_file   = pd.read_csv('machine-1-1_val.txt',   header=None, nrows=20000)

    train_data = torch.tensor(train_file.values, dtype=torch.float32)
    val_data   = torch.tensor(val_file.values, dtype=torch.float32)

    # Dataset
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len)
    val_dataset   = TimeSeriesDataset(val_data,   config.seq_len, config.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size)

    # Model
    net = model(
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_steps,
        eta_min=config.lr_min
    )

    # Evaluation
    @torch.no_grad()
    def estimate_loss():
        net.eval()
        losses = []

        for i, (xb, yb) in enumerate(val_loader):

            if i > 20:
                break

            xb, yb = xb.to(device), yb.to(device)

            pred = net(xb)
            pred = pred[:, -config.pred_len:, :]

            loss = F.mse_loss(pred, yb)
            losses.append(loss.item())

        net.train()
        return sum(losses) / len(losses)

    # Training
    best_val = float("inf")

    for epoch in range(config.n_epochs):
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

            if it > 300:  # for sweep
                break

            if it % config.eval_iter == 0:
                val_loss = estimate_loss()
                best_val = min(best_val, val_loss)

                print(f"step {it}: val loss {val_loss:.4f}")

    return best_val


def sweep_train():
    run = wandb.init()
    config = wandb.config
    val_loss = train(config)
    wandb.log({"val_loss": val_loss})

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="PeriodicSignalModel")
    wandb.agent(sweep_id, function=sweep_train, count=20)


















'''
def train(config):

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(config.eval_iter, device=device)
            for it, (xb, yb) in enumerate(loader):
                if it == eval_iter:
                    break

                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pred = pred[:, -config.pred_len:, :]

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
    train_dataset = TimeSeriesDataset(train_data, config.seq_len, config.pred_len)
    val_dataset = TimeSeriesDataset(config.val_data, config.seq_len, config.pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # create model
    model = model(config.n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, sigma, alpha)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_steps, eta_min=lr_min)

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
'''


