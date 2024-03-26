from lstm_pytorch import ConvLSTM, ConvGRU
import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, step, device, kernel_sizes, convType, layerIdx, isPeep):
        super(DownBlock, self).__init__()
        self.convType = convType
        if convType == 'LSTM':
            self.downConv = ConvLSTM(in_channels, out_channels, kernel_sizes=kernel_sizes, step=step, device=device,
                                     layer_index=layerIdx, isPeep=isPeep)
        elif convType == 'GRU':
            self.downConv = ConvGRU(in_channels, out_channels, kernel_sizes=kernel_sizes, step=step, device=device,
                                    layer_index=layerIdx)
        elif convType == 'Mix':
            self.downConv1 = ConvLSTM(in_channels, out_channels, kernel_sizes=kernel_sizes, step=step, device=device,
                                      layer_index=layerIdx)
            self.downConv = nn.Sequential(
                nn.Conv2d(out_channels[0], out_channels[0], kernel_sizes[0], 1, int((kernel_sizes[0] - 1) / 2)),
                nn.BatchNorm2d(out_channels[0]), nn.ReLU(inplace=False))
        else:
            self.downConv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_sizes[0], 1, int((kernel_sizes[0] - 1) / 2)),
                nn.BatchNorm2d(out_channels[0]), nn.ReLU(inplace=False),
                nn.Conv2d(out_channels[0], out_channels[0], kernel_sizes[0], 1, int((kernel_sizes[0] - 1) / 2)),
                nn.BatchNorm2d(out_channels[0]), nn.ReLU(inplace=False))
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, seq):
        if self.convType == 'LSTM' or self.convType == 'GRU':
            skip_out = self.downConv(x, seq)
        elif self.convType == 'Mix':
            x = self.downConv1(x, seq)
            skip_out = self.downConv(x)
        else:
            skip_out = self.downConv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(UpBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels - out_channels[0], in_channels - out_channels[0], kernel_size=2,
                                            stride=2)
        self.upConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_sizes, 1, int((kernel_sizes - 1) / 2)),
            nn.BatchNorm2d(out_channels[0]), nn.ReLU(inplace=False),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_sizes, 1, int((kernel_sizes - 1) / 2)),
            nn.BatchNorm2d(out_channels[0]), nn.ReLU(inplace=False))

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        up_out = self.upConv(x)
        return up_out


class UNet(nn.Module):
    def __init__(self, out_classes=1, filterList=[2, 16, 32, 64, 128], step=2, kernel_sizes=[3, 3, 3, 3, 3],
                 device='cuda'):
        super(UNet, self).__init__()
        self.filterSize = len(filterList)
        assert len(filterList) == len(kernel_sizes)

        # Downsampling Path
        for i in range(1, len(filterList) - 1):
            name = 'down_conv{}'.format(i)
            outListDown = []
            kernelListDown = []
            for _ in range(step):
                outListDown.append(filterList[i])
                kernelListDown.append(kernel_sizes[i - 1])
            setattr(self, name,
                    DownBlock(filterList[i - 1], outListDown, step, device=device, kernel_sizes=kernelListDown,
                              convType='Conv', layerIdx=i, isPeep=False))

        self.bottleConv = nn.Sequential(
            nn.Conv2d(filterList[-2], filterList[-1], kernel_sizes[-1], 1, int((kernel_sizes[-1] - 1) / 2)),
            nn.BatchNorm2d(filterList[-1]), nn.ReLU(inplace=False),
            nn.Conv2d(filterList[-1], filterList[-1], kernel_sizes[-1], 1, int((kernel_sizes[-1] - 1) / 2)),
            nn.BatchNorm2d(filterList[-1]), nn.ReLU(inplace=False))

        # Upsampling Path
        for i in range(len(filterList) - 2, 0, -1):
            name = 'up_conv{}'.format(i)
            outListUp = []
            for _ in range(step):
                outListUp.append(filterList[i])
            setattr(self, name, UpBlock(filterList[i] + filterList[i + 1], outListUp, kernel_sizes=kernel_sizes[i + 1]))

        # Final Convolution
        self.conv_last = nn.Conv2d(filterList[1], out_classes, kernel_size=3, stride=1, padding=1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        data = x
        for i in range(1, self.filterSize - 1):
            data, skip_out = getattr(self, 'down_conv{}'.format(i))(data, 0)
            setattr(self, 'skip_out{}'.format(i), skip_out)

        data = self.bottleConv(data)

        for i in range(self.filterSize - 2, 0, -1):
            data = getattr(self, 'up_conv{}'.format(i))(data, getattr(self, 'skip_out{}'.format(i)))

        data = self.conv_last(data)
        data = self.sigmoid_layer(data)
        x = data

        return x


class UNetRNN(nn.Module):
    def __init__(self, out_classes=1, filterList=[2, 16, 32, 64, 128], step=2, kernel_sizes=[3, 3, 3, 3, 3],
                 device='cuda', convType='LSTM', resolutionStage=1, isPeep=False):
        super(UNetRNN, self).__init__()
        self.filterSize = len(filterList)
        assert len(filterList) == len(kernel_sizes)

        # Downsampling Path
        for i in range(1, len(filterList) - 1):
            name = 'down_conv{}'.format(i)
            outListDown = []
            kernelListDown = []
            for _ in range(step):
                outListDown.append(filterList[i])
                kernelListDown.append(kernel_sizes[i - 1])
            self.convType = 'Conv'
            if i > resolutionStage:
                self.convType = convType
            setattr(self, name,
                    DownBlock(filterList[i - 1], outListDown, step, device=device, kernel_sizes=kernelListDown,
                              convType=self.convType, layerIdx=i, isPeep=isPeep))

        # Bottleneck
        if self.convType == 'LSTM':
            bottleList = step * [filterList[-1]]
            kernelList = step * [kernel_sizes[-1]]
            self.bottleConv = ConvLSTM(filterList[-2], bottleList, kernel_sizes=kernelList, step=step, device=device,
                                       layer_index=len(filterList) - 1, isPeep=isPeep)
        elif self.convType == 'Mix':
            bottleList = step * [filterList[-1]]
            kernelList = step * [kernel_sizes[-1]]
            self.bottleConv1 = ConvLSTM(filterList[-2], bottleList, kernel_sizes=kernelList, step=step, device=device,
                                        layer_index=len(filterList) - 1)
            self.bottleConv = nn.Sequential(
                nn.Conv2d(filterList[-1], filterList[-1], kernel_sizes[-1], 1, int((kernel_sizes[-1] - 1) / 2)),
                nn.BatchNorm2d(filterList[-1]), nn.ReLU(inplace=False))
        else:
            bottleList = step * [filterList[-1]]
            kernelList = step * [kernel_sizes[-1]]
            self.bottleConv = ConvGRU(filterList[-2], bottleList, kernel_sizes=kernelList, step=step, device=device,
                                      layer_index=len(filterList) - 1)

        # Upsampling Path
        for i in range(len(filterList) - 2, 0, -1):
            name = 'up_conv{}'.format(i)
            outListUp = []
            for _ in range(step):
                outListUp.append(filterList[i])
            setattr(self, name, UpBlock(filterList[i] + filterList[i + 1], outListUp, kernel_sizes=kernel_sizes[i + 1]))

        # Final Convolution
        self.conv_last = nn.Conv2d(filterList[1], out_classes, kernel_size=3, stride=1, padding=1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        outList = []
        for seq in range(x.shape[1]):
            data = x[:, seq, :, :, :]
            for i in range(1, self.filterSize - 1):
                data, skip_out = getattr(self, 'down_conv{}'.format(i))(data, seq)
                setattr(self, 'skip_out{}'.format(i), skip_out)

            if self.convType != 'Mix':
                data = self.bottleConv(data, seq)
            else:
                data = self.bottleConv1(data, seq)
                data = self.bottleConv(data)

            for i in range(self.filterSize - 2, 0, -1):
                data = getattr(self, 'up_conv{}'.format(i))(data, getattr(self, 'skip_out{}'.format(i)))

            data = self.conv_last(data)
            data = self.sigmoid_layer(data)
            outList.append(data)
            # x = [ (Nb, 1, Ny, Nx), (Nb, 1, Ny, Nx), ...]
        x = torch.cat(outList, dim=1)

        return x
