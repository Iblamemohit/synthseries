import torch
import matplotlib.pyplot as plt
import numpy as np
from synthseries import DiffusionProcess, TCNUNet1D, DDPMEngine

def generate_toy_data(num_samples=128, seq_len=256):
    """Generates a dataset of sine waves with varying frequencies and phases."""
    x = np.linspace(0, 10, seq_len)
    data = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        wave = np.sin(freq * x + phase) + np.random.normal(0, 0.1, seq_len)
        data.append(wave)
    # Shape: (Batch, Channels, SeqLen)
    return torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1)

def plot_forward_diffusion():
    print("Visualizing Forward Diffusion...")
    device = torch.device('cpu')
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)
    
    # Get a single toy sample
    x_start = generate_toy_data(1, 256)
    
    timesteps_to_plot = [0, 50, 150, 300, 600, 999]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Forward Diffusion Process (Adding Noise)', fontsize=16)
    
    for ax, t_val in zip(axes.flatten(), timesteps_to_plot):
        t = torch.tensor([t_val], device=device)
        x_noisy, _ = diffusion.add_noise(x_start, t)
        
        ax.plot(x_noisy.squeeze().numpy(), color='blue', alpha=0.7)
        ax.set_title(f'Timestep t={t_val}')
        ax.set_ylim(-3, 3)
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('forward_diffusion.png')
    print("Saved forward diffusion plot to forward_diffusion.png")

def train_and_sample():
    print("Training on Toy Dataset (Sine Waves) for a few epochs...")
    device = torch.device('cpu')
    
    # Tiny dataset and model for fast training
    dataset = generate_toy_data(256, 128)
    batch_size = 32
    seq_len = 128
    channels = 1
    
    diffusion = DiffusionProcess(num_timesteps=200, device=device) # fewer steps for speed
    model = TCNUNet1D(
        in_channels=channels,
        out_channels=channels,
        base_channels=32,
        channel_mults=(1, 2, 2)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    engine = DDPMEngine(model, diffusion, optimizer)
    
    epochs = 150
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size].to(device)
            loss = engine.step(batch)
            epoch_loss += loss
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / (len(dataset)/batch_size):.4f}")
            
    print("Generating new samples using Reverse Diffusion...")
    sampled_paths = diffusion.sample(model, shape=(4, channels, seq_len))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Synthetically Generated Time-Series (After Quick Training)', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(sampled_paths[i].squeeze().cpu().numpy(), color='green', alpha=0.8)
        ax.set_title(f'Synthetic Sample {i+1}')
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('synthetic_samples.png')
    print("Saved synthetic samples plot to synthetic_samples.png")

if __name__ == "__main__":
    plot_forward_diffusion()
    train_and_sample()
