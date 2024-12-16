import torch
from rvc.lib.algorithm.commons import fused_add_tanh_sigmoid_multiply, subsequent_mask
from rvc.lib.algorithm.normalization import LayerNorm
from rvc.lib.algorithm.attentions import FFN, MultiHeadAttention

class FFT(torch.nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.,
               proximal_bias=False, proximal_init=True, isflow = False, **kwargs):
    super().__init__()
    
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    
    if isflow:
        cond_layer = torch.nn.Conv1d(kwargs["gin_channels"], 2*hidden_channels*n_layers, 1)
        self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
        self.cond_layer = torch.nn.utils.parametrizations.weight_norm(cond_layer, name='weight')
        self.gin_channels = kwargs["gin_channels"]
    
    self.drop = torch.nn.Dropout(p_dropout)
    self.self_attn_layers = torch.nn.ModuleList()
    self.norm_layers_0 = torch.nn.ModuleList()
    self.ffn_layers = torch.nn.ModuleList()
    self.norm_layers_1 = torch.nn.ModuleList()
    
    for i in range(self.n_layers):
        self.self_attn_layers.append(
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
        self.norm_layers_0.append(LayerNorm(hidden_channels))
        self.ffn_layers.append(
            FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
        self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, g = None):
    """
    x: decoder input
    h: encoder output
    """
    if g is not None:
      g = self.cond_layer(g)

    self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    
    x = x * x_mask
    for i in range(self.n_layers):
        if g is not None:
            x = self.cond_pre(x)
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            x = fused_add_tanh_sigmoid_multiply(
                x,
                g_l,
                torch.IntTensor([self.hidden_channels]))
        y = self.self_attn_layers[i](x, x, self_attn_mask)
        y = self.drop(y)
        x = self.norm_layers_0[i](x + y)

        y = self.ffn_layers[i](x, x_mask)
        y = self.drop(y)
        x = self.norm_layers_1[i](x + y)
    x = x * x_mask
    return x

class F0Decoder(torch.nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = torch.nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = torch.nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = torch.nn.Conv1d(spk_channels, hidden_channels, 1)
    
    def _f0uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > 0.0)
        return uv

    def _normalize_f0(self, f0, x_mask, uv, random_scale=True):
        # calculate means based on x_mask
        uv_sum = torch.sum(uv, dim=1, keepdim=True)
        uv_sum[uv_sum == 0] = 9999
        means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

        if random_scale:
            factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
        else:
            factor = torch.ones(f0.shape[0], 1).to(f0.device)
        # normalize f0 based on means and factor
        f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
        if torch.isnan(f0_norm).any():
            exit(0)
        return f0_norm * x_mask
    
    def forward(self, pitch, m_p, x_mask, spk_emb=None):
        uv = self._f0uv(pitch)
        lf0 = 2595. * torch.log10(1. + pitch.unsqueeze(1) / 700.) / 500
        norm_lf0 = self._normalize_f0(lf0, x_mask, uv)
        
        #pred_lf0 = self.f0_decoder(m_p, norm_lf0, x_mask, spk_emb=g)
    
        x = m_p.detach()
        
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
            
        x += self.f0_prenet(norm_lf0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        pred_lf0 = self.proj(x) * x_mask
        return pred_lf0, norm_lf0, lf0