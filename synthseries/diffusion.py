import torch
import torch.nn as nn
from typing import Tuple, Optional

class DiffusionProcess:
    """
    Implements the Forward Acceleration Process and Reverse Denoising Loop
    for the Conditional Time-Series DDPM.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initializes the Diffusion Process with a linear beta schedule.
        
        Args:
            num_timesteps: Total number of diffusion steps (T).
            beta_start: Starting value for the beta schedule.
            beta_end: Ending value for the beta schedule.
            device: The device to place the scheduling tensors on.
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Shifted cumprod for reverse sampling
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ])

        # Pre-calculated values for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-calculated values for reverse diffusion
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extracts appropriate values from a 1D tensor `a` for a batch of indices `t`.
        Reshapes the extracted values to match the dimensions of `x_shape` for broadcasting.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The Forward Acceleration Process.
        Adds Gaussian noise to a batch of sequences at timestep `t` using the closed-form formulation.
        
        Args:
            x_start: The initial sequences (x_0). Shape: (Batch_Size, Channels, Sequence_Length)
            t: The timestep to sample at for each item in the batch. Shape: (Batch_Size,)
            noise: Optional pre-sampled noise. If None, samples from N(0, I).
            
        Returns:
            x_noisy: The noisy sequences (x_t). Shape: (Batch_Size, Channels, Sequence_Length)
            noise: The actual noise added. Shape: (Batch_Size, Channels, Sequence_Length)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        A single reverse denoising step for timestep `t_index`.
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict the noise using the model
        predicted_noise = model(x, t, cond)
        
        # Estimate mean using the predicted noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Log variance clipping to prevent instability when posterior variance is 0
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The complete Reverse Loop.
        Systematically denoises pure Gaussian noise back into a reconstructed 1D sequence.
        
        Args:
            model: The DDPM model (e.g., 1D-UNet backbone).
            shape: Shape of the tensor to generate (Batch_Size, Channels, Sequence_Length).
            cond: Optional conditioning tensor.
            
        Returns:
            x: The generated synthetic time-series. Shape: (Batch_Size, Channels, Sequence_Length)
        """
        device = self.device
        x = torch.randn(shape, device=device)
        batch_size = shape[0]

        model.eval()
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i, cond)
        model.train()
            
        return x
