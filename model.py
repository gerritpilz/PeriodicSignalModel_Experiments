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

    def forward(self, x):
        B,T,C = x.shape
        p, freq_bin = self.get_periods(x)  # amps, periods, freq_bin: (B, k)

        # helper
        to_int = lambda x: int(x.item())

        fin_timeframes=[]
        amps_list=[]
        f_off_list=[]
        for k in range(self.k_periods):
            nk_max = to_int(p[:, k].max())  # (1): max num columns of period k timeframes across batches
            mk_max = to_int(torch.ceil(T / p[:, k]).max())

            pad = (-T) % p[:, k]  # (B): pad for 1D time of period k for each batch
            n = p[:, k]           # (B): num columns of 2D timeframe of period k for each batch
            m = (T + pad) // n

            amp, f_off = self.bandpass(x, freq_bin[:, k])  # (B, T, C) each
            amps_list.append(amp) # append amp_t tensor to collect all k
            f_off_list.append(f_off)

            batch_timeframes = []
            for b in range(B):
                pad_b = pad[b]
                m_b = m[b]
                n_b = n[b]

                # shape/pad dims to create one tensor of 2D timeframes for each k
                timeframe_b_k = F.pad(x[b], (0, 0, 0,pad_b),) # (T+pad, C): pad time dim

                # 2D transform
                timeframe_b_k = rearrange(timeframe_b_k, '(m n) c -> m n c', n=n_b)
                timeframe_b_k = F.pad(timeframe_b_k, (0,0, 0,nk_max - n_b, 0,mk_max - m_b))  # pad to match dims across batches for fixed k

                batch_timeframes.append(timeframe_b_k) # (M_max, N_max, C)

            timeframes_k = torch.stack(batch_timeframes, dim=0) # (B, M_max, N_max C)

            # Conv returns (B,M_max,N_max,C)
            timeframes_k = self.times_conv(timeframes_k)

            # flatten to 1D
            fin_timeframes_k = []
            for b in range(B):
                timeframe_b_k = timeframes_k[b, :to_int(m[b]), :to_int(n[b]), :]  # trunc away pads for M_max/N_max
                timeframe_b_k = rearrange(timeframe_b_k, 'm n c -> (m n) c')  # flatten to 1D
                timeframe_b_k = timeframe_b_k[:T, :] # trunc away padded time: (T, C)
                fin_timeframes_k.append(timeframe_b_k)

            fin_timeframes.append(torch.stack(fin_timeframes_k, dim=0)) # (B, T, C)

        timeframes = torch.stack(fin_timeframes, dim=1) # (B, k, T, C)
        amps = torch.stack(amps_list, dim=1)           # (B, k, T, C)
        f_off = torch.stack(f_off_list, dim=1)

        # MLP
        x = self.MLP(timeframes)

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
        x = x + x_film
        '''

        # Adaptive Aggregation
        weights = F.softmax(amps, dim=1)         # (B, k, T, C) -> softmax across k, importance of freq k at each time/channel
        x_weighted = x * weights  # (B, k, T, C)
        x = x_weighted.sum(dim=1)
        dx = rearrange(x_weighted, 'b k t c -> b t (k c)')
        dx = self.agg_MLP(dx)      # (B T k*C) -> (B T C); learn cross-period dependencies
        out = x + dx

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
        f0_bin = rearrange(f0_bin, 'b -> b 1') # add freq dim
        H = (torch.abs(f_bin - f0_bin) <= self.bw).float()  # (B 1) * (1 F) = (B F)
        return H


    def gaussian_bandpass(self, f0_bin, sigma=1): #sigma 1 optimal
        f_bin = torch.arange(self.seq_len // 2 + 1, device='cuda')  # freq bin vector
        f0_bin = rearrange(f0_bin, 'b -> b 1')  # add freq dim
        d = torch.abs(f_bin - f0_bin)
        H = torch.exp(-(d ** 2) / (2 * sigma ** 2))
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
        Hf = self.Hf_bandpass(f0_bin) # (B F)´  #!!!!!!!!!!!!!!!!!!!!!!!!!!! sigma umstellen
        Hf = rearrange(Hf, 'b f -> b f 1')
        Xf_filt = Hf*Xf
        x_filt = torch.fft.irfft(Xf_filt, dim=-2) # (B T C)

        # hilbert transform the filtered x
        z = self.analytic_signal(x_filt) # (B T C)

        amp_t = torch.abs(z)

        phase_t = self.unwrap(torch.angle(z), dim=-2)  # angle returns [-pi, pi] -> unwrap
        freq_t = torch.diff(phase_t, dim=-2) / (2.0 * torch.pi)  # (B T C)
        freq_t = F.pad(freq_t, (0, 0, 0, 1))  # add lost time step
        f0 = rearrange(f0_bin/self.seq_len, 'b -> b 1 1')
        freq_offset = f0 - freq_t

        return amp_t, freq_offset # both (B T C)


    def get_periods(self, x):
        T = x.shape[1]

        x_ft = torch.fft.rfft(x, dim=-2)
        x_ft = x_ft[:, 1:, :]  # drop row 0 -> const term
        amps = torch.abs(x_ft)
        amps = torch.mean(amps, dim=-1)  # avg across channel dim (identify most meaningful periods across channels) -> (B,F)
        amps_k, freq_bin_k = torch.topk(amps, k=self.k_periods,dim=-1)  # top k_periods amps, frequencies(indices): (B, k) each
        periods_k = T // (freq_bin_k + 1)  # row 0 = freq 0 sliced out before -> new row 0 refers to freq 1 -> shift freq_k by one
        return periods_k, freq_bin_k + 1


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

    def forward(self, input):
        x = self.embd(input)
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

