import torch
from typing import Optional
from rvc.lib.algorithm.nsf import GeneratorNSF
from rvc.lib.algorithm.generators import Generator
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.residuals import ResidualCouplingBlock
from rvc.lib.algorithm.encoders import TextEncoder, PosteriorEncoder
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
        MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init))
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

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0.half())
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x
def normalize_f0(f0, x_mask, uv, random_scale=True):
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
class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model.

    Args:
        spec_channels (int): Number of channels in the spectrogram.
        segment_size (int): Size of the audio segment.
        inter_channels (int): Number of channels in the intermediate layers.
        hidden_channels (int): Number of channels in the hidden layers.
        filter_channels (int): Number of channels in the filter layers.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the encoder.
        kernel_size (int): Size of the convolution kernel.
        p_dropout (float): Dropout probability.
        resblock (str): Type of residual block.
        resblock_kernel_sizes (list): Kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for the residual blocks.
        upsample_rates (list): Upsampling rates for the decoder.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes for the upsampling layers.
        spk_embed_dim (int): Dimension of the speaker embedding.
        gin_channels (int): Number of channels in the global conditioning vector.
        sr (int): Sampling rate of the audio.
        use_f0 (bool): Whether to use F0 information.
        text_enc_hidden_dim (int): Hidden dimension for the text encoder.
        kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        auto_f0_prediction: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.use_f0 = use_f0
        self.auto_f0_prediction = auto_f0_prediction

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )

        if use_f0:
            self.dec = GeneratorNSF(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
                is_half=kwargs["is_half"],
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        if auto_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def _remove_weight_norm_from(self, module):
        """Utility to remove weight normalization from a module."""
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self
    
    def _f0uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > 0.0)
        return uv

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_lengths: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            y (torch.Tensor, optional): Target spectrogram.
            y_lengths (torch.Tensor, optional): Lengths of the target spectrograms.
            ds (torch.Tensor, optional): Speaker embedding.
        """
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)

            if self.use_f0 and pitchf is not None:
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g)
            else:
                o = self.dec(z_slice, g=g)
                
            if self.auto_f0_prediction:
                uv = self._f0uv(pitch)
                lf0 = 2595. * torch.log10(1. + pitch.unsqueeze(1) / 700.) / 500
                norm_lf0 = normalize_f0(lf0, x_mask, uv)
                pred_lf0 = self.f0_decoder(m_p, norm_lf0, x_mask, spk_emb=g)
            else:
                lf0 = 0
                norm_lf0 = 0
                pred_lf0 = 0

            return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0

        return None, None, x_mask, None, (None, None, m_p, logs_p, None, None), None, None, None

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
        f0_prediction: bool = False,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        
        if f0_prediction:
            uv = self._f0uv(pitch)
            lf0 = 2595. * torch.log10(1. + pitch.unsqueeze(1) / 700.) / 500.
            norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(m_p, norm_lf0, x_mask, g)
            nsff0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = (
            self.dec(z * x_mask, nsff0, g=g)
            if self.use_f0
            else self.dec(z * x_mask, g=g)
        )

        return o, x_mask, (z, z_p, m_p, logs_p)
