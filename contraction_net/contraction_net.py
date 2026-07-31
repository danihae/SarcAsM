import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedBlock(nn.Module):
    """Residual dilated-convolution block: Conv(k=5, dilation=d) -> BN -> GELU -> Conv(1x1)."""

    def __init__(self, n_filter, dilation, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_filter, n_filter, kernel_size=5,
                              padding=2 * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(n_filter)
        self.point = nn.Conv1d(n_filter, n_filter, kernel_size=1)
        self.bn_point = nn.BatchNorm1d(n_filter)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = F.gelu(self.bn(self.conv(x)))
        h = self.dropout(h)
        h = self.bn_point(self.point(h))
        return F.gelu(x + h)


class ContractionNet(nn.Module):
    """
    Dilated temporal convolutional network for contraction detection.

    Two properties drive the design.

    **No statistic is shared across time.** Every normalisation is
    :class:`~torch.nn.BatchNorm1d`, which at inference is a fixed per-channel affine
    transform with *no* dependence on the trace being analysed. This is what keeps the
    answer independent of how much of a recording is spent contracting. Any normalisation
    that pools over the time axis destroys that property: it makes the majority state the
    implicit zero, so in a recording that is mostly contracting the decision inverts.
    :class:`~torch.nn.InstanceNorm1d` does this by construction, and
    :class:`~torch.nn.GroupNorm` is not an escape -- PyTorch normalises it over the spatial
    axis too. Amplitude invariance is instead handled once, explicitly, in
    :func:`~contraction_net.prediction.prepare_robust_input`.

    **Context spans three orders of magnitude of temporal scale.** The dilated residual
    stack is translation-equivariant and has no length-extrapolation term. With
    ``dilations=(1, 2, 4, 8, 16, 32, 64)`` the receptive field is 513 frames while the
    innermost blocks still resolve detail a few frames wide, so one model covers both a
    contraction resolved by 3 samples and one spread over 500. Reaching long range with a
    single global attention layer would not do: without positional encoding it is
    permutation-equivariant and can only contribute an order-blind summary of the whole
    trace -- which is itself a duty prior -- and its magnitude drifts with sequence length.

    Parameters
    ----------
    n_filter : int, optional
        Channel width throughout the trunk. Default is 64.
    in_channels : int, optional
        Number of input channels. Default is 2 (level and per-frame difference).
    out_channels : int, optional
        Number of output channels. Default is 3: contraction state, onset and offset.
    dropout_rate : float, optional
        Dropout applied inside each block. Default is 0.1. Keep it modest, and note that
        anything above zero makes predictions non-deterministic unless the module is in
        ``eval()`` mode -- which :func:`~contraction_net.prediction.predict_contractions`
        guarantees.
    dilations : sequence of int, optional
        Dilation of each residual block. Default is ``(1, 2, 4, 8, 16, 32, 64)``.

    Notes
    -----
    Output channels 1 and 2 predict contraction **boundaries** -- onsets give the beating
    rate, onset/offset pairs give the cycle duration, which is what the analysis actually
    consumes. Unlike a segmentation mask a boundary is a local event, so predicting one
    does not require knowing how much of the recording is spent contracting.
    """

    def __init__(self, n_filter=64, in_channels=2, out_channels=3, dropout_rate=0.1,
                 dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.stem = nn.Conv1d(in_channels, n_filter, kernel_size=5, padding=2)
        self.stem_bn = nn.BatchNorm1d(n_filter)
        self.blocks = nn.ModuleList(
            [_DilatedBlock(n_filter, d, dropout_rate) for d in self.dilations])
        self.head = nn.Conv1d(n_filter, out_channels, kernel_size=1)

    @property
    def receptive_field(self):
        """Frames of context the trunk sees, counting the stem and every block."""
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
            The corresponding logits, for use with loss functions that expect them.
        """
        h = F.gelu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return torch.sigmoid(logits), logits
