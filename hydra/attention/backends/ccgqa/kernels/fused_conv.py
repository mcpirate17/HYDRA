"""
Optimized convolution for CCGQA.
Uses PyTorch's native Conv1d but with better memory layout and fusion opportunities.
"""

import torch
import torch.nn as nn


class OptimizedConvSequence(nn.Module):
    """
    Optimized sequential convolution for CCGQA.
    Combines conv1 + conv2 with better memory access patterns.
    """
    
    def __init__(self, channels, groups1, groups2, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # First conv (depthwise)
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,  # We'll handle padding manually for causal
            groups=groups1,
            bias=False,
        )
        
        # Second conv (pointwise or full)
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups2,
            bias=False,
        )
    
    def forward(self, x):
        """
        Apply causal convolution sequence.
        Input: [B, N, C]
        Output: [B, N, C]
        """
        B, N, C = x.shape
        
        # Transpose to [B, C, N] for Conv1d
        x = x.transpose(1, 2).contiguous()
        
        # Apply conv1 with causal padding
        pad = self.kernel_size - 1
        x = nn.functional.pad(x, (pad, 0))
        x = self.conv1(x)
        
        # Apply conv2 with causal padding
        x = nn.functional.pad(x, (pad, 0))
        x = self.conv2(x)
        
        # Transpose back to [B, N, C]
        x = x.transpose(1, 2).contiguous()
        
        return x


def replace_conv_sequence(conv1, conv2):
    """
    Replace two Conv1d modules with optimized sequence.
    Copies weights from existing modules.
    """
    channels = conv1.out_channels
    groups1 = conv1.groups
    groups2 = conv2.groups
    kernel_size = conv1.kernel_size[0]
    
    opt_conv = OptimizedConvSequence(channels, groups1, groups2, kernel_size)
    
    # Copy weights
    opt_conv.conv1.weight.data.copy_(conv1.weight.data)
    opt_conv.conv2.weight.data.copy_(conv2.weight.data)
    
    return opt_conv

