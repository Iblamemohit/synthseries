import torch
import torch.nn as nn
import math
from typing import Optional

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Injects timestep 't' into the network blocks.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timestep tensor of shape (Batch_Size,)
        Returns:
            Embeddings of shape (Batch_Size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualTCNBlock(nn.Module):
    """
    1D Dilated Temporal Convolution block with residual connections.
    Avoids sequential bottlenecks by processing sequences concurrently.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dilation: int = 1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Dilated causal/standard convolutions. Here we use standard "same" padding for U-Net compatibility 
        # so sequence length is preserved.
        padding = dilation  # Kernel size 3, padding = dilation * (kernel_size - 1) // 2
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (Batch_Size, In_Channels, Sequence_Length)
            time_emb: Time embeddings, shape (Batch_Size, Time_Emb_Dim)
            
        Returns:
            Output tensor, shape (Batch_Size, Out_Channels, Sequence_Length)
        """
        # First conv block
        h = self.block1(x)
        
        # Add time embedding (requires reshaping to broadcast over sequence length)
        time_emb = self.mlp(time_emb).unsqueeze(-1)
        h = h + time_emb
        
        # Second conv block
        h = self.block2(h)
        
        # Residual connection
        return h + self.residual_conv(x)

class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halves the sequence length."""
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Doubles the sequence length."""
        return self.conv(x)

class TCNUNet1D(nn.Module):
    """
    1D-UNet Backbone configured for Time-Series (Batch, Channels, Sequence_Length).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        time_emb_dim: Optional[int] = None,
        cond_channels: int = 0
    ):
        """
        Args:
            in_channels: The number of input channels (1 for univariate time-series).
            out_channels: The number of output channels (1 to predict noise).
            base_channels: Base number of filters.
            channel_mults: Multiplier for channels at each down/up resolution.
            time_emb_dim: Dimension for time embeddings. Defaults to base_channels * 4.
            cond_channels: Number of channels for optional conditioning.
        """
        super().__init__()
        
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else base_channels * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        self.init_conv = nn.Conv1d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        # Downsampling path
        for i, mult in enumerate(channel_mults):
            out_dim = base_channels * mult
            is_last = (i == len(channel_mults) - 1)
            
            self.downs.append(nn.ModuleList([
                ResidualTCNBlock(now_channels, out_dim, self.time_emb_dim, dilation=1),
                ResidualTCNBlock(out_dim, out_dim, self.time_emb_dim, dilation=2),
                Downsample1D(out_dim, out_dim) if not is_last else nn.Identity()
            ]))
            now_channels = out_dim
            channels.append(now_channels)
            
        # Bottleneck
        self.mid_block1 = ResidualTCNBlock(now_channels, now_channels, self.time_emb_dim, dilation=4)
        self.mid_block2 = ResidualTCNBlock(now_channels, now_channels, self.time_emb_dim, dilation=8)
        
        # Upsampling path
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_dim = base_channels * mult
            is_last = (i == 0)
            skip_channels = channels.pop()
            
            self.ups.append(nn.ModuleList([
                ResidualTCNBlock(now_channels + skip_channels, out_dim, self.time_emb_dim, dilation=2),
                ResidualTCNBlock(out_dim, out_dim, self.time_emb_dim, dilation=1),
                Upsample1D(out_dim, out_dim) if not is_last else nn.Identity()
            ]))
            now_channels = out_dim
            
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv1d(now_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (Batch_Size, Channels, Sequence_Length).
            time: Timesteps for each batch item, shape (Batch_Size,).
            cond: Optional conditional sequence/tensor, shape (Batch_Size, Cond_Channels, Sequence_Length).
            
        Returns:
            Predicted noise tensor of shape (Batch_Size, Channels, Sequence_Length).
        """
        if cond is not None:
            # Concatenate condition along channel dimension
            x = torch.cat([x, cond], dim=1)
            
        x = self.init_conv(x)
        t = self.time_mlp(time)
        
        skip_connections = [x]
        
        # Down
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skip_connections.append(x)
            x = downsample(x)
            
        # Mid
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Up
        for block1, block2, upsample in self.ups:
            skip_x = skip_connections.pop()
            # Concatenate skip connection along channel dimension
            x = torch.cat([x, skip_x], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            
        return self.final_conv(x)
