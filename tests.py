import torch
from synthseries import DiffusionProcess, TCNUNet1D, DDPMEngine

def test_synthseries():
    batch_size = 4
    channels = 1
    seq_len = 256
    
    device = torch.device('cpu')
    
    print("Testing Diffusion Process...")
    diffusion = DiffusionProcess(num_timesteps=100, device=device)
    
    x_start = torch.randn(batch_size, channels, seq_len)
    t = torch.randint(0, 100, (batch_size,))
    
    x_noisy, noise = diffusion.add_noise(x_start, t)
    assert x_noisy.shape == (batch_size, channels, seq_len)
    assert noise.shape == (batch_size, channels, seq_len)
    
    print("Testing TCN-UNet1D...")
    model = TCNUNet1D(
        in_channels=channels,
        out_channels=channels,
        base_channels=32,
        channel_mults=(1, 2, 4)
    )
    
    predicted_noise = model(x_noisy, t)
    assert predicted_noise.shape == (batch_size, channels, seq_len)
    
    print("Testing Engine Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine = DDPMEngine(model, diffusion, optimizer)
    
    loss = engine.step(x_start)
    print(f"Step Loss: {loss}")
    
    print("Testing Reverse Sampling Loop...")
    sampled_paths = diffusion.sample(model, shape=(batch_size, channels, seq_len))
    assert sampled_paths.shape == (batch_size, channels, seq_len)
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_synthseries()
