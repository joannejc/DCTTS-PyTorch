import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class NonCausalConv1d(nn.Module):
    """Non causal Conv1d with appropriate padding to ensure sequence length stays the same.

    Note Convolutions always have stride of 1 following layout in paper.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size-1)*dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding, dilation)


    def forward(self, x):
        """
        Inputs:
            x(batch_size x input_dim x seq_len)

        """
        return self.conv(x)


class CausalConv1d(nn.Module):
    """Causal conv1d with appropriate padding to ensure sequence length stays the same.

    Note Convolutions always have stride of 1 following layout in paper.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1, self.padding, dilation)

    def forward(self, x):
        """
        Inputs:
            x(batch_size x input_dim x seq_len)
        
        """
        x = self.conv(x)
        # left padding for causal
        if self.padding > 0:
            return x[:,:,:-self.padding].contiguous()
        else:
            return x


class Highway(nn.Module):
    """Highway network with conv1d
    
    """
    def __init__(self, hidden, kernel_size, dilation, causal=False):
        self.d = hidden
        super().__init__()
        if causal:
            self.conv = CausalConv1d(hidden, 2*hidden, kernel_size, dilation)
        else:
            self.conv = NonCausalConv1d(hidden, 2*hidden, kernel_size, dilation)
        
    def forward(self, x):
        """
        Inputs:
            x(batch_size x input_dim x seq_len)
        
        """
        Hout = self.conv(x)
        H1 = Hout[:,:self.d,:]
        H2 = Hout[:,self.d:,:]
        return F.sigmoid(H1)*H2 + (1-F.sigmoid(H1))*x




def test_module_shapes():
    conv1 = NonCausalConv1d(256,256*2,3,3)
    conv2 = CausalConv1d(256, 256*2, 3, 9)
    hc = Highway(256,3,3,True)
    out = conv1(torch.rand(5,256,40))
    out2 = conv2(torch.rand(5,256,40))
    out3 = hc(torch.rand(5,256,80))
    assert out.shape[1] == 256*2 and out.shape[2] == 40
    assert out2.shape[1] == 256*2 and out2.shape[2] == 40
    assert out3.shape[1] == 256 and out3.shape[2] == 80
