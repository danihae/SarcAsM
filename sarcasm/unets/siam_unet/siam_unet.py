import torch
from torch import nn
import torch.nn.functional as F
import logging


class Siam_UNet(nn.Module):
    """Implementation of Siamese U-Net inspired by Dunnhofer et al. https://doi.org/10.1016/j.media.2019.101631"""
    def __init__(self, n_filter=32, mode='max'):
        super().__init__()
        # mode for combining T-1 and T
        self.mode = mode
        # encode
        self.encode1 = self.conv(1, n_filter)
        self.encode2 = self.conv(n_filter, n_filter)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode3 = self.conv(n_filter, 2 * n_filter)
        self.encode4 = self.conv(2 * n_filter, 2 * n_filter)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode5 = self.conv(2 * n_filter, 4 * n_filter)
        self.encode6 = self.conv(4 * n_filter, 4 * n_filter)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode7 = self.conv(4 * n_filter, 8 * n_filter)
        self.encode8 = self.conv(8 * n_filter, 8 * n_filter, dropout=0.5)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # middle
        self.middle_conv1 = self.conv(8 * n_filter, 16 * n_filter)
        self.middle_conv2 = self.conv(16 * n_filter, 16 * n_filter, dropout=0.5)

        # decode
        self.up1 = nn.ConvTranspose2d(16 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
        self.decode1 = self.conv(16 * n_filter, 8 * n_filter)
        self.decode2 = self.conv(8 * n_filter, 8 * n_filter)
        self.up2 = nn.ConvTranspose2d(8 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
        self.decode3 = self.conv(8 * n_filter, 4 * n_filter)
        self.decode4 = self.conv(4 * n_filter, 4 * n_filter)
        self.up3 = nn.ConvTranspose2d(4 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        self.decode5 = self.conv(4 * n_filter, 2 * n_filter)
        self.decode6 = self.conv(2 * n_filter, 2 * n_filter)
        self.up4 = nn.ConvTranspose2d(2 * n_filter, 1 * n_filter, kernel_size=2, stride=2)
        self.decode7 = self.conv(2 * n_filter, 1 * n_filter)
        self.decode8 = self.conv(1 * n_filter, 1 * n_filter)
        self.decode9 = self.conv(1 * n_filter, 1)
        self.final = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0),
        )

    def conv(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        return block

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            # logging.critical(x1.shape, x2.shape)
            logging.critical(f"Shapes: {x1.shape}, {x2.shape}")
            raise ValueError('concatenation failed: wrong dimensions')

    def depthwise_xcorr(self, embed_curr, embed_prev):
        """depth-wise cross correlation"""
        batch = embed_prev.size(0)
        channel = embed_prev.size(1)
        embed_curr = embed_curr.view(1, batch * channel, embed_curr.size(2), embed_curr.size(3))
        embed_prev = embed_prev.view(batch * channel, 1, embed_prev.size(2), embed_prev.size(3))
        out = F.conv2d(embed_curr, embed_prev, groups=batch * channel, padding='same')
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, x, prev_x):
        # top encoder (current frame)
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        m1 = self.maxpool1(e2)
        e3 = self.encode3(m1)
        e4 = self.encode4(e3)
        m2 = self.maxpool2(e4)
        e5 = self.encode5(m2)
        e6 = self.encode6(e5)
        m3 = self.maxpool2(e6)
        e7 = self.encode7(m3)
        e8 = self.encode8(e7)
        m4 = self.maxpool2(e8)

        # bottom encoder (previous frame)
        me1 = self.encode1(prev_x)
        me2 = self.encode2(me1)
        mm1 = self.maxpool1(me2)
        me3 = self.encode3(mm1)
        me4 = self.encode4(me3)
        mm2 = self.maxpool2(me4)
        me5 = self.encode5(mm2)
        me6 = self.encode6(me5)
        mm3 = self.maxpool2(me6)
        me7 = self.encode7(mm3)
        me8 = self.encode8(me7)
        mm4 = self.maxpool2(me8)

        # depth-wise cross-correlation or element-wise maximum
        if self.mode == 'corr':
            corr = self.depthwise_xcorr(m4, mm4)
        elif self.mode == 'max':
            corr = torch.maximum(m4, mm4)
        # mid layer
        mid1 = self.middle_conv1(corr)
        mid2 = self.middle_conv2(mid1)

        # decoder
        u1 = self.up1(mid2)
        c1 = self.concat(u1, e7)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)
        u2 = self.up2(d2)
        c2 = self.concat(u2, e5)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)
        u3 = self.up3(d4)
        c3 = self.concat(u3, e3)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)
        u4 = self.up4(d6)
        c4 = self.concat(u4, e1)
        d7 = self.decode7(c4)
        d8 = self.decode8(d7)
        d9 = self.decode9(d8)
        logits = self.final(d9)
        return torch.sigmoid(logits), logits
