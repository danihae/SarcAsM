import torch
import torch.nn as nn
import torch.nn.functional as F


class ContractionNet(nn.Module):
    def __init__(self, n_filter=64, in_channels=1, out_channels=2, dropout_rate=0.5):
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
