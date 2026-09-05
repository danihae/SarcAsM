import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ContractionNet', 'SymmetrizedContractionNet', 'NORMS',
           'ATTENTION_POSITIONS']

#: Normalisation variants selectable for ablation. Only ``'batch'`` is free of any
#: statistic pooled over time; the others couple the answer to the duty cycle.
NORMS = ('batch', 'instance', 'group', 'none')

#: Where a global attention layer may sit relative to the dilated stack. ``'pre'`` gives
#: the stack an already-contextualised input; ``'post'`` summarises what it produced;
#: ``'mid'`` splits the difference. ``False`` omits it.
ATTENTION_POSITIONS = (False, 'pre', 'mid', 'post', 'both')


def _make_norm(kind, n_filter):
    if kind == 'batch':
        return nn.BatchNorm1d(n_filter)
    if kind == 'instance':
        return nn.InstanceNorm1d(n_filter, affine=True)
    if kind == 'group':
        return nn.GroupNorm(min(8, n_filter), n_filter)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f'norm must be one of {NORMS}, got {kind!r}')


class _DilatedBlock(nn.Module):
    """Residual block: Conv(k=5, dilation=d) -> norm -> GELU -> Conv(1x1) -> norm."""

    def __init__(self, n_filter, dilation, dropout_rate=0.1, norm='batch'):
        super().__init__()
        self.conv = nn.Conv1d(n_filter, n_filter, kernel_size=5,
                              padding=2 * dilation, dilation=dilation)
        self.bn = _make_norm(norm, n_filter)
        self.point = nn.Conv1d(n_filter, n_filter, kernel_size=1)
        self.bn_point = _make_norm(norm, n_filter)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = F.gelu(self.bn(self.conv(x)))
        h = self.dropout(h)
        h = self.bn_point(self.point(h))
        return F.gelu(x + h)


class ContractionNet(nn.Module):
    """
    Dilated temporal convolutional network for contraction detection.

    BatchNorm keeps every normalisation independent of the trace being analysed, so the
    answer does not depend on how much of a recording is spent contracting; amplitude
    invariance is handled once in
    :func:`~contraction_net.prediction.prepare_robust_input`. The dilated residual stack
    is translation-equivariant and reaches 513 frames of context while its innermost
    blocks still resolve a few frames.

    Parameters
    ----------
    n_filter : int, optional
        Channel width throughout the trunk. Default is 64.
    in_channels : int, optional
        Input channels. Default is 2 (level and per-frame difference).
    out_channels : int, optional
        Output channels. Default is 3: contraction state, onset, offset.
    dropout_rate : float, optional
        Dropout inside each block. Default is 0.1. Non-zero dropout makes predictions
        non-deterministic outside ``eval()`` mode.
    dilations : sequence of int, optional
        Dilation per residual block. Default is ``(1, 2, 4, 8, 16, 32, 64)``.
    norm : str, optional
        One of :data:`NORMS`. Default is ``'batch'``. The others exist for ablation.
    attention : bool or str, optional
        Position of a global self-attention layer; one of :data:`ATTENTION_POSITIONS`.
        ``True`` means ``'post'``. Default is False.

    Notes
    -----
    Channels 1 and 2 predict boundaries: onsets give the beating rate and onset/offset
    pairs the cycle duration, which is what the analysis consumes.
    """

    def __init__(self, n_filter=64, in_channels=2, out_channels=3, dropout_rate=0.1,
                 dilations=(1, 2, 4, 8, 16, 32, 64), norm='batch', attention=False):
        super().__init__()
        self.dilations = tuple(dilations)
        self.norm_kind = norm
        self.stem = nn.Conv1d(in_channels, n_filter, kernel_size=5, padding=2)
        self.stem_bn = _make_norm(norm, n_filter)
        self.blocks = nn.ModuleList(
            [_DilatedBlock(n_filter, d, dropout_rate, norm) for d in self.dilations])
        position = 'post' if attention is True else attention
        if position not in ATTENTION_POSITIONS:
            raise ValueError(f'attention must be one of {ATTENTION_POSITIONS}, '
                             f'got {attention!r}')
        self.attn_position = position
        if position:
            self.attn = nn.MultiheadAttention(n_filter, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(n_filter)
            if position == 'both':
                self.attn2 = nn.MultiheadAttention(n_filter, num_heads=4, batch_first=True)
                self.attn_norm2 = nn.LayerNorm(n_filter)
        else:
            self.attn = None
        self.head = nn.Conv1d(n_filter, out_channels, kernel_size=1)

    def _attend(self, h, layer=None, norm=None):
        layer = self.attn if layer is None else layer
        norm = self.attn_norm if norm is None else norm
        t = h.transpose(1, 2)
        a, _ = layer(t, t, t)
        return norm(a + t).transpose(1, 2)

    @property
    def receptive_field(self):
        """Frames of context the trunk sees, counting the stem and every block."""
        if self.attn is not None:
            return None
        return 1 + 4 + sum(4 * d for d in self.dilations)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_channels, length)``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(batch, out_channels, length)``.
        torch.Tensor
            The corresponding logits.
        """
        h = F.gelu(self.stem_bn(self.stem(x)))
        pos = self.attn_position
        if pos in ('pre', 'both'):
            h = self._attend(h)
        mid = len(self.blocks) // 2 - 1
        for i, block in enumerate(self.blocks):
            h = block(h)
            if pos == 'mid' and i == mid:
                h = self._attend(h)
        if pos == 'post':
            h = self._attend(h)
        elif pos == 'both':
            h = self._attend(h, self.attn2, self.attn_norm2)
        logits = self.head(h)
        return torch.sigmoid(logits), logits


class SymmetrizedContractionNet(nn.Module):
    """``f(x) = reduce(h(x), h(-x))`` over one shared trunk.

    Polarity invariance holds exactly, by construction, rather than being learned from
    sign-flip augmentation. Both polarities go through the trunk in a single batch so
    BatchNorm sees the same statistics for each; running them as two calls would make
    the invariance hold only in ``eval()`` mode.

    Parameters
    ----------
    reduce : str, optional
        ``'max'`` or ``'mean'``. Default is ``'max'``.

    Other parameters are forwarded to :class:`ContractionNet`.
    """

    def __init__(self, n_filter=64, in_channels=2, out_channels=3, dropout_rate=0.1,
                 dilations=(1, 2, 4, 8, 16, 32, 64), norm='batch', attention=False,
                 reduce='max'):
        super().__init__()
        if reduce not in ('max', 'mean'):
            raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
        self.trunk = ContractionNet(n_filter, in_channels, out_channels, dropout_rate,
                                    dilations, norm=norm, attention=attention)
        self.reduce = reduce

    @property
    def receptive_field(self):
        return self.trunk.receptive_field

    def forward(self, x):
        _, logits = self.trunk(torch.cat([x, -x], 0))
        a, b = logits.chunk(2, 0)
        logits = torch.maximum(a, b) if self.reduce == 'max' else 0.5 * (a + b)
        return torch.sigmoid(logits), logits
