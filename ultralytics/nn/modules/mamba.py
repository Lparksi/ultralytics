import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba3 import Mamba3

from ultralytics.nn.modules.conv import Conv


class C2Mamba(nn.Module):
    """C2Mamba module integrating Mamba3 for global context extraction."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initialize C2Mamba module.
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Mamba3 blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        # Mamba3 modules
        self.m = nn.ModuleList(
            Mamba3(
                d_model=self.c,
                d_state=16, # Adjust based on needs / parameters
                expand=2,
                headdim=64 if self.c % 64 == 0 else min(32, self.c),
            ) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2Mamba."""
        B, C, H, W = x.shape
        
        # Split into two branches
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # Branch b goes through Mamba3
        # Mamba3 expects sequence input: (B, L, D) where L = H*W
        b = rearrange(b, 'b c h w -> b (h w) c')
        
        for mamba_block in self.m:
            b = mamba_block(b)
            
        b = rearrange(b, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Concatenate and pass through final conv
        return self.cv2(torch.cat((a, b), 1))
