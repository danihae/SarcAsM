import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvGRUCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wz = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uz = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wr = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Ur = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wh = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uh = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

    def forward(self, x, h):
        zt = torch.sigmoid(self.Wz(x) + self.Uz(h))
        rt = torch.sigmoid(self.Wr(x) + self.Ur(h))
        ht_hat = torch.tanh(self.Wh(x) + self.Uh(rt * h))
        ht = (1 - zt) * h + zt * ht_hat
        return ht

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device))


class PeepConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(PeepConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uho = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Vi = None
        self.Vf = None
        self.Vo = None


def forward(self, x, h, c):
    it = torch.sigmoid(self.Wxi(x) + self.Uhi(h) + c * self.Vi)
    ft = torch.sigmoid(self.Wxf(x) + self.Uhf(h) + c * self.Vf)
    ct = ft * c + it * torch.tanh(self.Wxc(x) + self.Uhc(h))
    ot = torch.sigmoid(self.Wxo(x) + self.Uho(h) + ct * self.Vo)
    ht = ot * torch.tanh(ct)
    return ht, ct


def init_hidden(self, batch_size, hidden, shape):
    if self.Vi is None:
        self.Vi = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
        self.Vf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
        self.Vo = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(self.device)
    else:
        assert shape[0] == self.Vi.size()[2], 'Input Height Mismatched!'
        assert shape[1] == self.Vi.size()[3], 'Input Width Mismatched!'
    return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device),
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uhc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Uho = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

    def forward(self, x, h, c):
        it = torch.sigmoid(self.Wxi(x) + self.Uhi(h))
        ft = torch.sigmoid(self.Wxf(x) + self.Uhf(h))
        ct = ft * c + it * torch.tanh(self.Wxc(x) + self.Uhc(h))
        ot = torch.sigmoid(self.Wxo(x) + self.Uho(h))
        ht = ot * torch.tanh(ct)
        return ht, ct


def init_hidden(self, batch_size, hidden, shape):
    return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device),
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(self.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_sizes, step=1, device='cuda', layer_index=1,
                 isPeep=False):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels[:-1]
        self.hidden_channels = hidden_channels
        self.layer_index = layer_index

        self.step = step

        assert len(self.input_channels) == len(self.hidden_channels) == len(kernel_sizes) == self.step

        self.relu_layer = nn.ReLU(inplace=False)
        self.sigmoid_layer = nn.Sigmoid()
        self.conv_last = nn.Conv2d(self.hidden_channels[-1], 1, kernel_size=3, stride=1, padding=1)
        for i in range(self.step):
            if isPeep:
                cell = PeepConvLSTMCell(self.input_channels[i], self.hidden_channels[i], kernel_sizes[i], device)
            else:
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], kernel_sizes[i], device)
            setattr(self, 'cell{}_{}'.format(i, self.layer_index), cell)
            setattr(self, 'batchN{}_{}'.format(i, self.layer_index), nn.BatchNorm2d(hidden_channels[i]))

    def forward(self, input, seq):
        x = input
        for i in range(self.step):

            name = 'cell{}_{}'.format(i, self.layer_index)

            if seq == 0:
                bsize, _, height, width = x.size()
                (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                         shape=(height, width))

            if seq != 0:
                h, c = getattr(self, 'h_{}_{}'.format(i, self.layer_index)), getattr(self, 'c_{}_{}'.format(i,
                                                                                                            self.layer_index))

            new_h, new_c = getattr(self, name)(x.float(), h.float(), c.float())
            x = new_h
            x = self.relu_layer(x)
            x = getattr(self, 'batchN{}_{}'.format(i, self.layer_index))(x)
            setattr(self, 'h_{}_{}'.format(i, self.layer_index), new_h)
            setattr(self, 'c_{}_{}'.format(i, self.layer_index), new_c)

        return x


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_sizes, step=1, device='cuda', layer_index=1):
        super(ConvGRU, self).__init__()
        self.input_channels = [input_channels] + hidden_channels[:-1]
        self.hidden_channels = hidden_channels
        self.layer_index = layer_index

        self.step = step

        assert len(self.input_channels) == len(self.hidden_channels) == len(kernel_sizes) == self.step

        self.relu_layer = nn.ReLU(inplace=False)
        self.sigmoid_layer = nn.Sigmoid()
        self.conv_last = nn.Conv2d(self.hidden_channels[-1], 1, kernel_size=3, stride=1, padding=1)
        for i in range(self.step):
            cell = ConvGRUCell(self.input_channels[i], self.hidden_channels[i], kernel_sizes[i], device)
            setattr(self, 'cell{}_{}'.format(i, self.layerHere))
            setattr(self, 'batchN{}_{}'.format(i, self.layer_index), nn.BatchNorm2d(hidden_channels[i]))

    def forward(self, input, seq):
        x = input
        for i in range(self.step):

            name = 'cell{}_{}'.format(i, self.layer_index)

            if seq == 0:
                bsize, _, height, width = x.size()
                (h) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                      shape=(height, width))

            if seq != 0:
                h = getattr(self, 'h_{}_{}'.format(i, self.layer_index))
            new_h = getattr(self, name)(x.float(), h.float())
            x = new_h
            x = self.relu_layer(x)
            x = getattr(self, 'batchN{}_{}'.format(i, self.layer_index))(x)
            setattr(self, 'h_{}_{}'.format(i, self.layer_index), new_h)

        return x
