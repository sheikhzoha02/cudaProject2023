import torch
import torch.nn as nn
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size, _, height, width = x.size()
            hidden = (
                torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device))
        hx, cx = hidden
        combined = torch.cat([x, hx], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        cy = f * cx + i * g
        hy = o * torch.tanh(cy)
        return hy, cy