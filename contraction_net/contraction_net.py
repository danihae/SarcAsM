import torch
import torch.nn as nn
import torch.nn.functional as F


class ContractionNet(nn.Module):
    """
        ContractionNet model for detecting contraction intervals from time-series data of individual Z-band positions
        and sarcomere lengths of beating cardiomyocytes.

        This neural network is designed to handle noisy data and distinguish between contracting and non-contracting intervals.
        The network first extracts various features from a single input time-series by two convolutional layers with kernel size 5, followed by a dilated convolution in the third layer
        to capture broader temporal patterns. Each convolution is followed by instance normalization and ReLU activation.
        A self-attention layer enhances focus on salient features. The processed signal then undergoes two further
        convolutions before being outputted through a sigmoid activation function.

        Methods
        -------
        forward(x)
            Forward pass through the network.
        """
    def __init__(self, n_filter=64, in_channels=1, out_channels=2, dropout_rate=0.5):
        """
        Parameters
        ----------
        n_filter : int, optional
            Number of filters in the convolutional layers (default is 64).
        in_channels : int, optional
            Number of input channels (default is 1).
        out_channels : int, optional
            Number of output channels (default is 2).
        dropout_rate : float, optional
            Dropout rate (default is 0.5).
        """
        super(ContractionNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=5, padding=2)
        self.in1 = nn.InstanceNorm1d(n_filter)
        self.conv2 = nn.Conv1d(in_channels=n_filter, out_channels=n_filter * 2, kernel_size=5, padding=2)
        self.bn2 = nn.InstanceNorm1d(n_filter * 2)
        self.conv3 = nn.Conv1d(in_channels=n_filter * 2, out_channels=n_filter * 4, kernel_size=5, padding=4,
                               dilation=2)
        self.bn3 = nn.InstanceNorm1d(n_filter * 4)
        self.attention = nn.MultiheadAttention(embed_dim=n_filter * 4, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(n_filter * 4)
        self.dropout_attention = nn.Dropout(dropout_rate)
        self.conv4 = nn.Conv1d(in_channels=n_filter * 4, out_channels=n_filter * 2, kernel_size=5, padding=2,
                               dilation=1)
        self.bn4 = nn.InstanceNorm1d(n_filter * 2)
        self.dropout_pre_output = nn.Dropout(dropout_rate)
        self.conv_out = nn.Conv1d(in_channels=n_filter * 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length) after sigmoid activation.
        torch.Tensor
            Raw output tensor of shape (batch_size, out_channels, sequence_length).
        """
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        residual = x
        x, attention_weights = self.attention(x, x, x)
        x = x + residual
        x = self.norm1(x)
        x = self.dropout_attention(x)
        x = x.transpose(1, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_pre_output(x)
        x = self.conv_out(x)
        return torch.sigmoid(x), x


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


class ContractionNetV2(nn.Module):
    """
    Dilated temporal convolutional network for contraction detection.

    Replaces :class:`ContractionNet`, whose two measured failure modes both trace back to
    its structure rather than to its training:

    **Duty-cycle coupling.** Every normalisation layer in the original is
    ``nn.InstanceNorm1d``, which takes statistics over the whole time axis. Once most of a
    recording is spent contracting, "contracted" becomes the normalised zero and the
    decision inverts -- mean IoU falls from 0.90 below duty 0.65 to 0.66 above 0.75. This
    was confirmed causally: feeding the same weights the same trace but with normalisation
    statistics borrowed from a low-duty reference lifted IoU from 0.32 to 0.93.
    Here every normalisation is :class:`~torch.nn.BatchNorm1d`, which at inference is a
    fixed per-channel affine transform with *no* dependence on the trace being analysed.
    Note that :class:`~torch.nn.GroupNorm` would not have helped: PyTorch normalises it over
    the spatial axis too, reintroducing exactly the coupling being removed.

    **Range of temporal scale.** The original resolves 21 frames of context through its
    convolutions; its only long-range path is a single self-attention layer with no
    positional encoding, which is permutation-equivariant (verified numerically) and so
    carries no temporal order at all -- it can only contribute an order-blind global
    summary, which is itself a duty prior. Its output magnitude also drifts with sequence
    length, and it was trained at one fixed length. Here the attention is gone and context
    comes from a stack of dilated residual blocks, translation-equivariant with no
    length-extrapolation term. With ``dilations=(1, 2, 4, 8, 16, 32, 64)`` the receptive
    field is 513 frames, while the innermost blocks still resolve detail a few frames wide
    -- the span needed to cover both a contraction resolved by 3 samples and one spread
    over 500. The original scored IoU 0.00 at 3 frames per contraction and needed ~20 to
    work at all.

    Parameters
    ----------
    n_filter : int, optional
        Channel width throughout the trunk. Default is 64.
    in_channels : int, optional
        Number of input channels. Default is 2 (level and velocity).
    out_channels : int, optional
        Number of output channels. Default is 3: contraction state, onset and offset.
    dropout_rate : float, optional
        Dropout applied inside each block. Default is 0.1, far below the original's 0.5 --
        which, combined with the missing ``eval()`` call at inference, was making every
        prediction a random ensemble member.
    dilations : sequence of int, optional
        Dilation of each residual block. Default is ``(1, 2, 4, 8, 16, 32, 64)``.

    Notes
    -----
    The third output channel pair predicts contraction **boundaries** rather than the
    original's per-event-normalised distance transform. Boundaries are what the analysis
    actually consumes -- onsets give the beating rate, onset/offset pairs give the cycle
    duration -- and unlike a segmentation mask a boundary is a local event, so predicting it
    does not require knowing how much of the recording is spent contracting.
    """

    #: Input convention this architecture expects; see ``prediction.predict_contractions``.
    input_norm = 'robust'

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
