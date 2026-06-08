import torch
import pandas as pd
import argparse
from times_model import times_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(data_path, checkpoint_path, output_path):
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']

    # load model
    net = times_model(
        config['n_channels'],
        config['seq_len'],
        config['d_embd'],
        config['dropout'],
        config['n_timeBlocks'],
        config['k_periods'],
        config['sigma'],
        config['alpha']
    ).to(device)
    net.load_state_dict(checkpoint['model_state'])
    net.eval()

    # load data
    df = pd.read_csv(data_path, header=None)
    data = torch.tensor(df.values, dtype=torch.float32)

    seq_len  = config['seq_len']
    pred_len = config['pred_len']

    # use last seq_len timesteps as input
    x = data[-seq_len:].unsqueeze(0).to(device)  # (1, seq_len, n_channels)

    # inference
    with torch.no_grad():
        pred = net(x)
        pred = pred[:, -pred_len:, :]  # (1, pred_len, n_channels)

    # save
    pred_np = pred.squeeze(0).cpu().numpy()
    pd.DataFrame(pred_np).to_csv(output_path, index=False, header=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       required=True, help='Path to input file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output',     required=True, help='Path to output file')
    args = parser.parse_args()

    predict(args.data, args.checkpoint, args.output)