import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class MLP(nn.Module):
     def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2 * d_out),
            nn.GELU(),
            nn.Linear(2 * d_out, d_out),
            nn.Dropout(dropout)
         )

     def forward(self, x):
          return self.net(x)

class MLP_film(nn.Module):
    def __init__(self, d_in, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2*d_in),
            nn.GELU(),
            nn.Linear(2*d_in, 2*d_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TimesBlockConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.conv_list = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2),
            nn.Conv2d(d_model, d_model, kernel_size=7, padding=3),
        ])
        self.proj = nn.Conv2d(len(self.conv_list) * d_model, d_model, kernel_size=1)

        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(d_model)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        outs = [conv(x) for conv in self.conv_list]
        out = torch.cat(outs, dim=1)

        out = self.proj(out)
        out = self.bn(out)
        out = self.activation(out)

        out = out + x

        out = out.permute(0, 2, 3, 1)
        return out


class block(nn.Module):
    def __init__(self, seq_len, d_embd, dropout, k_periods, bw, n_heads):
        super().__init__()
        self.MLP = MLP(d_embd, d_embd,dropout)
        self.MLP_film = MLP_film(d_embd, dropout)
        self.times_conv = TimesBlockConv(d_embd)
        self.ln = nn.LayerNorm(d_embd)
        self.attention =  nn.MultiheadAttention(embed_dim=d_embd, num_heads=n_heads, batch_first=True)
        self.agg_MLP = MLP(d_embd*k_periods, d_embd, dropout)

        self.d_embd = d_embd
        self.seq_len = seq_len
        self.k_periods = k_periods
        self.bw = bw

        # hilbert filter
        H = torch.zeros(seq_len)
        H[0] = 1
        H[1:seq_len // 2] = 2
        H[seq_len // 2] = 1
        self.register_buffer("hilbert_filter", H)

    def forward(self, x, eval=False):
        B,T,C = x.shape
        x_in = x

        # top k frequencies
        periods, freq_bins, amps_k = self.get_periods(x)  # amps, periods, freq_bin

        x_list=[]
        amps_list=[]
        f_off_list=[]
        for k in range(self.k_periods):
            # Instant amp and freq
            amp, f_off = self.bandpass(x, freq_bins[k])  # (B, T, C) each
            amps_list.append(amp)  # append amp_t tensor to collect all k
            f_off_list.append(f_off)

            # pad dims for 2D
            pad = (-T) % periods[k]     # pad for 1D time of period k
            n = periods[k]         # num columns of 2D timeframe of period k

            # 2D reshape
            x_k = F.pad(x,(0, 0, 0, pad))  # (B T+pad C)
            x_k = rearrange(x_k, 'b (m n) c -> b m n c', n=n.item())

            # Conv
            x_k = self.times_conv(x_k)

            # flatten 1D
            x_k = rearrange(x_k, 'b m n c -> b (m n) c')
            x_k = x_k[:, :T, :]  # trunc away padded time
            x_list.append(x_k)

        x = torch.stack(x_list, dim=1)        # (B, k, T, C) each
        amps = torch.stack(amps_list, dim=1)
        f_off = torch.stack(f_off_list, dim=1)

        # MLP
        x = self.MLP(x)

        '''
        # Attention optional
        B = x.shape[0]
        x_att = self.ln(x)
        x_att = rearrange(x_att, 'b k t c -> (b k) t c')
        x_att, _ = self.attention(x_att, x_att, x_att)
        x_att = rearrange(x_att, '(b k) t c -> b k t c', b=B)
        x = x + x_att
        '''

        '''
        # Film
        param = self.MLP_film(f_off)
        x_film = x*param[..., :self.d_embd] + param[..., self.d_embd:]
        xf = x + x_film
        '''


        # Adaptive Aggregation
        amps = self.MLP(amps)
        weights = F.softmax(amps, dim=1)         # (B, k, T, C) -> softmax across k, importance of freq k at each time/channel
        x_weighted = x * weights  # (B, k, T, C)
        x_weighted = x_weighted.sum(dim=1)
        #dx = rearrange(x_weighted, 'b k t c -> b t (k c)')
        #dx = self.agg_MLP(dx)      # (B T k*C) -> (B T C); learn cross-period dependencies
        out = x_in + x_weighted

        '''
        # Aggregation original
        weights = F.softmax(amps_k)
        weights = rearrange(weights, 'k -> k 1 1')
        x = x * weights
        out = x.sum(1)
        '''

        return out

    def analytic_signal(self, x):
        H = self.hilbert_filter
        H = rearrange(H, 't -> t 1')
        Xf = torch.fft.fft(x, dim=-2)
        Zf = Xf * H
        z = torch.fft.ifft(Zf, dim=-2)
        return z

    def Hf_bandpass(self, f0_bin):
        f_bin = torch.arange(self.seq_len//2 + 1, device='cuda') # freq bin vector
        H = (torch.abs(f_bin - f0_bin) <= self.bw).float()  # (F)
        return H


    def gaussian_bandpass(self, f0_bin, sigma=1): #sigma 1 optimal
        f_bin = torch.arange(self.seq_len // 2 + 1, device='cuda')  # freq bin vector
        d = torch.abs(f_bin - f0_bin)
        H = torch.exp(-(d ** 2) / (2 * sigma ** 2)) # (F)
        return H

    def unwrap(self, phase, dim):
        dphi = torch.diff(phase, dim=dim)

        # bring into [-pi, pi]
        dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi

        phase_unwrapped = torch.cumsum(
            torch.cat([phase.narrow(dim, 0, 1), dphi], dim=dim),
            dim=dim
        )
        return phase_unwrapped

    def bandpass(self, x, f0_bin):

        Xf = torch.fft.rfft(x, dim=-2) # (B F C)
        Hf = self.Hf_bandpass(f0_bin) # (B F)  #!!!!!!!!!!!!!!!!!!!!!!!!!!! sigma umstellen
        Hf = rearrange(Hf, 'f -> f 1')
        Xf_filt = Hf*Xf
        x_filt = torch.fft.irfft(Xf_filt, dim=-2) # (B T C)

        # hilbert transform the filtered x
        z = self.analytic_signal(x_filt) # (B T C)

        amp_t = torch.abs(z)

        phase_t = self.unwrap(torch.angle(z), dim=-2)  # angle returns [-pi, pi] -> unwrap
        freq_t = torch.diff(phase_t, dim=-2) / (2.0 * torch.pi)  # (B T C)
        freq_t = F.pad(freq_t, (0, 0, 0, 1))  # add lost time step
        f0 = f0_bin/self.seq_len
        freq_offset = f0 - freq_t

        return amp_t, freq_offset # both (B T C)


    def get_periods(self, x):
        T = x.shape[1]

        X_f = torch.fft.rfft(x, dim=-2)
        X_f = X_f[:, 1:, :]  # drop row 0 -> const term
        amps = torch.abs(X_f)
        amps = torch.mean(amps, dim=(0,-1))  # avg across B,C
        amps_k, freq_bin_k = torch.topk(amps, k=self.k_periods)  # top k amps, freq_bins
        freq_bin_k = freq_bin_k + 1 # row 0 = freq 0 sliced out before -> new row 0 refers to freq 1 -> shift freq_k by one
        periods_k = T // freq_bin_k
        return periods_k, freq_bin_k, amps_k


class model(nn.Module):
    def __init__(self, n_channels, seq_len, d_embd, dropout, n_timeBlocks, k_periods, bw, n_heads):
        super().__init__()

        self.embd = nn.Linear(n_channels, d_embd)
        self.blocks = nn.Sequential(*[block(seq_len, d_embd, dropout, k_periods, bw, n_heads) for _ in range(n_timeBlocks)])
        self.embd_back = nn.Linear(d_embd, n_channels)

        self.seq_len = seq_len

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input, eval=False):
        x = self.embd(input)

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-5)

        x = self.blocks(x)
        pred = self.embd_back(x)  # (B, T, C)
        return pred

    def generate(self, context, max_new_pred):   #  context: (B, T_context, C)
        for _ in range(max_new_pred):
            # crop idx to last seq_len embeddings
            context_cond = context[:, -self.seq_len:, :]

            # get predictions
            pred, loss = self(context_cond)

            # only last time step
            pred_next = pred[:, [-1], :]  # (B, 1, C)

            # append pred_t to running sequence
            context = torch.concat([context, pred_next], dim=1)

        return context

