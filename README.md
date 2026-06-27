# SynthSeries: Time-Series Denoising Diffusion Probabilistic Model (DDPM)

**SynthSeries** is a high-performance, production-grade Generative AI framework designed for synthesizing financial asset paths using Conditional Time-Series Denoising Diffusion Probabilistic Models (DDPM).

Built entirely on PyTorch, it leverages a 1D-UNet backbone with Dilated Temporal Convolutions (TCN) to avoid sequential execution bottlenecks, enabling massively parallel processing on GPUs.

## Features
- **Continuous Forward Process:** Closed-form continuous Gaussian noise injection without slow iteration loops.
- **1D-UNet Backbone:** Custom-built Neural Network for sequences of shape `(Batch, Channels, Sequence_Length)` utilizing Residual Dilated Temporal blocks instead of legacy LSTMs.
- **Sinusoidal Positional Embeddings:** Accurate timestep embeddings injected directly into intermediate network states.
- **Interactive Streamlit Dashboard:** An interactive web UI to rapidly tune diffusion parameters (timesteps, beta variance) and neural architecture depths while monitoring MSE loss live.
- **Out-of-the-box GBM Simulation:** Pre-built Geometric Brownian Motion simulator for rapid prototyping of generative stress-testing.

## Installation

Ensure you have Python 3.9+ and PyTorch installed. Then, install the required dependencies:

```bash
pip install torch numpy matplotlib streamlit
```

Clone the repository:
```bash
git clone https://github.com/Iblamemohit/synthseries.git
cd synthseries
```

## Running the Interactive Dashboard

To launch the hyperparameter tuning interface and visualize synthetic generations in real-time, launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Code Structure

- `synthseries/diffusion.py`: Contains the `DiffusionProcess` class responsible for defining the noise schedules and managing the forward acceleration / reverse sampling loops.
- `synthseries/models/tcn_unet.py`: Defines the `TCNUNet1D` backbone architecture and the temporal convolution layers.
- `synthseries/training/engine.py`: Encapsulates the `DDPMEngine` responsible for calculating the Mean Squared Error (MSE) objective against the true noise distribution and executing gradient descent steps.
- `dashboard.py`: The entrypoint for the interactive hyperparameter tuning application.
- `visualize.py`: Script to generate static `matplotlib` visualizations for the forward diffusion process and reverse synthetic outputs.
- `tests.py`: Sanity checks validating tensor dimensions, gradient propagation, and shape adherence.

## Basic Usage

You can embed SynthSeries directly into your data pipelines:

```python
import torch
from synthseries import DiffusionProcess, TCNUNet1D, DDPMEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the Diffusion Process
diffusion = DiffusionProcess(num_timesteps=200, beta_end=0.02, device=device)

# 2. Construct the 1D-UNet Backbone
model = TCNUNet1D(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    channel_mults=(1, 2, 4)
).to(device)

# 3. Set up the Training Engine
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
engine = DDPMEngine(model, diffusion, optimizer)

# 4. Train on your stationary log-returns dataset (Batch, 1, Seq_Len)
for epoch in range(100):
    loss = engine.step(your_tensor_batch)

# 5. Generate fully synthetic financial paths from pure noise
synthetic_log_returns = diffusion.sample(model, shape=(10, 1, 256))
```

## Disclaimer
This model is designed for research, algorithmic stress-testing, and quant engineering workflows. Synthetic asset generation should not be used as the sole basis for live trading models.
