import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from synthseries.diffusion import DiffusionProcess

class DDPMEngine:
    """
    Training Engine for the Conditional Time-Series DDPM.
    Computes the loss minimizing the Mean Squared Error (MSE) between 
    the true noise injected and the noise predicted by the 1D-UNet.
    """
    def __init__(self, model: nn.Module, diffusion: DiffusionProcess, optimizer: optim.Optimizer):
        """
        Args:
            model: The DDPM Backbone (e.g., TCNUNet1D).
            diffusion: The DiffusionProcess logic (contains beta/alpha schedules).
            optimizer: The optimizer (e.g., AdamW) for updating the model.
        """
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def compute_loss(self, x_start: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the MSE loss for a batch of data.
        
        Args:
            x_start: The true, noiseless sequences (x_0). Shape: (Batch_Size, Channels, Sequence_Length).
            cond: Optional conditional information. Shape: (Batch_Size, Cond_Channels, Sequence_Length).
            
        Returns:
            Scalar tensor representing the MSE loss.
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 1. Randomly sample a timestep 't' for each sample in the batch
        t = torch.randint(
            0, self.diffusion.num_timesteps, (batch_size,), device=device, dtype=torch.long
        )
        
        # 2. Sample noise to add to the sequences
        noise = torch.randn_like(x_start)
        
        # 3. Add noise to x_start to create the noisy x_t
        x_noisy, true_noise = self.diffusion.add_noise(x_start, t, noise)
        
        # 4. Predict the noise using the neural network
        predicted_noise = self.model(x_noisy, t, cond)
        
        # 5. Compute the MSE loss between true noise and predicted noise
        loss = self.criterion(predicted_noise, true_noise)
        
        return loss
        
    def step(self, x_start: torch.Tensor, cond: Optional[torch.Tensor] = None) -> float:
        """
        Performs a single training step: computes loss, backpropagates, and updates weights.
        
        Args:
            x_start: True sequences (x_0).
            cond: Optional conditions.
            
        Returns:
            The loss value as a float.
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x_start, cond)
        loss.backward()
        
        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
