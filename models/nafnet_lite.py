import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Simple Gate
# ============================================================

class SimpleGate(nn.Module):
    """
    Splits the feature channels into two halves
    and multiplies them element-wise.

    Input:
        [B, 2C, H, W]

    Output:
        [B, C, H, W]
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ============================================================
# 2. NAF Block
# ============================================================

class NAFBlockLite(nn.Module):
    """
    Lightweight NAF-style restoration block.

    Structure:

        Input
          |
        GroupNorm
          |
        1x1 Conv
          |
      SimpleGate
          |
    Depthwise Conv
          |
        1x1 Conv
          |
      Residual
          |
        GroupNorm
          |
        1x1 Conv
          |
      SimpleGate
          |
        1x1 Conv
          |
      Residual
    """

    def __init__(self, channels):
        super().__init__()

        # ----------------------------------------------------
        # First transformation
        # ----------------------------------------------------

        self.norm1 = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
        )

        self.pw1 = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=1
        )

        self.sg = SimpleGate()

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

        self.pw2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Second transformation / FFN
        # ----------------------------------------------------

        self.norm2 = nn.GroupNorm(
            num_groups=1,
            num_channels=channels
        )

        self.ffn1 = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=1
        )

        self.ffn_gate = SimpleGate()

        self.ffn2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Learnable residual scaling
        # ----------------------------------------------------

        self.beta = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.gamma = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x):

        # ====================================================
        # Branch 1
        # ====================================================

        y = self.norm1(x)

        y = self.pw1(y)

        y = self.sg(y)

        y = self.dw(y)

        y = self.pw2(y)

        # Residual connection
        x = x + self.beta * y

        # ====================================================
        # Branch 2
        # ====================================================

        y = self.norm2(x)

        y = self.ffn1(y)

        y = self.ffn_gate(y)

        y = self.ffn2(y)

        # Residual connection
        x = x + self.gamma * y

        return x


# ============================================================
# 3. NAFNet-Lite
# ============================================================

class NAFNetLite(nn.Module):
    """
    Lightweight image restoration / super-resolution network.

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]

    Pipeline:

        LR Input
           |
        2x Bilinear Upsampling
           |
        Intro Conv
           |
        Encoder
           |
        Middle
           |
        Decoder
           |
        Ending Conv
           |
        Residual Output
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=32,
        enc_blocks=(2, 2, 4),
        middle_blocks=4,
        dec_blocks=(2, 2, 2),
        scale=2
    ):
        super().__init__()

        self.scale = scale

        # ====================================================
        # Input projection
        # ====================================================

        self.intro = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            padding=1
        )

        # ====================================================
        # Encoder
        # ====================================================

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width

        for num_blocks in enc_blocks:

            # Restoration blocks
            encoder = nn.Sequential(
                *[
                    NAFBlockLite(channels)
                    for _ in range(num_blocks)
                ]
            )

            self.encoders.append(encoder)

            # Downsampling
            self.downs.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2
                )
            )

            channels *= 2

        # ====================================================
        # Middle blocks
        # ====================================================

        self.middle = nn.Sequential(
            *[
                NAFBlockLite(channels)
                for _ in range(middle_blocks)
            ]
        )

        # ====================================================
        # Decoder
        # ====================================================

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for num_blocks in dec_blocks:

            # Upsampling:
            #
            # channels
            #    ↓
            # channels * 2
            #    ↓
            # PixelShuffle
            #    ↓
            # channels / 2
            #

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=1
                    ),
                    nn.PixelShuffle(2)
                )
            )

            channels //= 2

            # Decoder blocks
            decoder = nn.Sequential(
                *[
                    NAFBlockLite(channels)
                    for _ in range(num_blocks)
                ]
            )

            self.decoders.append(decoder)

        # ====================================================
        # Output projection
        # ====================================================

        self.ending = nn.Conv2d(
            width,
            out_channels,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Save original LR input
        # ----------------------------------------------------

        if self.scale != 1:

            inp = F.interpolate(
                x,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False
            )

        else:
            inp = x

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        x = self.intro(inp)

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        skips = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):

            x = encoder(x)

            # Save skip connection
            skips.append(x)

            # Downsample
            x = down(x)

        # ----------------------------------------------------
        # Middle
        # ----------------------------------------------------

        x = self.middle(x)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        for up, decoder, skip in zip(
            self.ups,
            self.decoders,
            reversed(skips)
        ):

            # Upsample
            x = up(x)

            # Safety check for spatial dimensions
            if x.shape[-2:] != skip.shape[-2:]:

                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Skip connection
            x = x + skip

            # Decoder blocks
            x = decoder(x)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output = self.ending(x)

        # ----------------------------------------------------
        # Global residual connection
        # ----------------------------------------------------

        output = output + inp

        return output


# ============================================================
# 4. Model Builder
# ============================================================

def build_model(cfg):
    """
    Builds NAFNetLite from YAML configuration.
    """

    return NAFNetLite(

        in_channels=cfg.get(
            "in_channels",
            1
        ),

        out_channels=cfg.get(
            "out_channels",
            1
        ),

        width=cfg.get(
            "width",
            32
        ),

        enc_blocks=tuple(
            cfg.get(
                "enc_blocks",
                [2, 2, 4]
            )
        ),

        middle_blocks=cfg.get(
            "middle_blocks",
            4
        ),

        dec_blocks=tuple(
            cfg.get(
                "dec_blocks",
                [2, 2, 2]
            )
        ),

        scale=cfg.get(
            "scale",
            2
        )
    )