
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
        )
    def forward(self, x): 
        return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.Conv2d(c_in, c_in, 3, stride=2, padding=1, groups=c_in)  # depthwise downsample
        self.block = ConvBlock(c_in, c_out)
    def forward(self, x): 
        return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c_in, c_out, 1)
        self.block = ConvBlock(c_in, c_out)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([self.conv1x1(x), skip], dim=1)
        return self.block(x)

class UNetSmall(nn.Module):
    """
    Input: [B,1,H,W]
    Output: [B,2,H,W]  (ER, NR) non-negative via Softplus
    """
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = Down(base, base*2)
        self.enc3 = Down(base*2, base*4)
        self.enc4 = Down(base*4, base*8)

        self.bott = ConvBlock(base*8, base*8)

        self.up3 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2)
        self.up1 = Up(base*2, base)

        self.head = nn.Conv2d(base, 2, 1)
        self.softplus = nn.Softplus(beta=1.0)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None: 
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.enc4(s3)
        x  = self.bott(x)
        x  = self.up3(x, s3)
        x  = self.up2(x, s2)
        x  = self.up1(x, s1)
        raw = self.head(x)
        return self.softplus(raw)
