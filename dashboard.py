import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from synthseries.diffusion import DiffusionProcess
from synthseries.models.tcn_unet import TCNUNet1D
from synthseries.training.engine import DDPMEngine

def generate_gbm_paths(num_paths=64, seq_len=128, mu=0.0, sigma=0.02, S0=1.0):
    """Generates Geometric Brownian Motion paths as a toy financial dataset."""
    dt = 1.0 / seq_len
    paths = np.zeros((num_paths, seq_len))
    paths[:, 0] = S0
    for t in range(1, seq_len):
        Z = np.random.standard_normal(num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # We will model the log returns, which are stationary, to make it easier for DDPM
    log_returns = np.diff(np.log(paths), axis=1)
    
    # Pad first element with zeros to keep sequence length
    log_returns = np.pad(log_returns, ((0,0), (1,0)), mode='constant')
    
    # Normalize to zero mean and unit variance for the neural net
    mean = np.mean(log_returns)
    std = np.std(log_returns)
    normalized_returns = (log_returns - mean) / (std + 1e-8)
    
    tensor_data = torch.tensor(normalized_returns, dtype=torch.float32).unsqueeze(1)
    return tensor_data, mean, std

st.set_page_config(page_title="SynthSeries DDPM", layout="wide")

st.title("SynthSeries: Conditional Time-Series DDPM")
st.markdown("Interactive Hyperparameter Tuning and Visualization for Synthetic Financial Asset Paths.")

with st.sidebar:
    st.header("⚙️ Hyperparameters")
    
    st.subheader("Diffusion Process")
    num_timesteps = st.slider("Timesteps (T)", min_value=50, max_value=1000, value=200, step=50)
    beta_end = st.slider("Beta End", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
    
    st.subheader("Model Architecture")
    base_channels = st.selectbox("Base Channels", options=[16, 32, 64], index=1)
    
    st.subheader("Training Parameters")
    epochs = st.slider("Epochs", min_value=10, max_value=500, value=100, step=10)
    learning_rate = st.selectbox("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], index=2)
    batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)

# Main Area
dataset, data_mean, data_std = generate_gbm_paths(num_paths=128, seq_len=128, sigma=0.1)

st.header("1. Training Dataset (Normalized Log Returns)")
fig, ax = plt.subplots(figsize=(10, 3))
for i in range(5):
    ax.plot(dataset[i].squeeze().numpy(), alpha=0.7)
ax.set_title("Sample Real Paths")
st.pyplot(fig)

if "model" not in st.session_state:
    st.session_state.model = None
if "diffusion" not in st.session_state:
    st.session_state.diffusion = None

if st.button("🚀 Train Model"):
    device = torch.device("cpu")
    
    st.session_state.diffusion = DiffusionProcess(num_timesteps=num_timesteps, beta_end=beta_end, device=device)
    
    st.session_state.model = TCNUNet1D(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        channel_mults=(1, 2, 2)
    ).to(device)
    
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=learning_rate)
    engine = DDPMEngine(st.session_state.model, st.session_state.diffusion, optimizer)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle dataset
        indices = torch.randperm(dataset.shape[0])
        shuffled_dataset = dataset[indices]
        
        for i in range(0, len(shuffled_dataset), batch_size):
            batch = shuffled_dataset[i:i+batch_size].to(device)
            loss = engine.step(batch)
            epoch_loss += loss
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Update UI every few epochs
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
            fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
            ax_loss.plot(losses, color='orange')
            ax_loss.set_title("Training Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("MSE Loss")
            loss_chart.pyplot(fig_loss)
            plt.close(fig_loss)

    st.success("Training Complete!")

if st.session_state.model is not None:
    st.header("2. Generate Synthetic Paths")
    if st.button("✨ Sample from DDPM"):
        with st.spinner("Running reverse diffusion..."):
            sampled_returns = st.session_state.diffusion.sample(
                st.session_state.model, 
                shape=(5, 1, 128)
            )
            
            fig_ret, ax_ret = plt.subplots(figsize=(10, 4))
            for i in range(5):
                ax_ret.plot(sampled_returns[i].squeeze().cpu().numpy(), alpha=0.8, label=f"Path {i+1}")
            ax_ret.set_title("Synthetic Normalized Log Returns")
            ax_ret.legend()
            st.pyplot(fig_ret)
            
            # Reconstruct prices
            st.subheader("Reconstructed Price Paths")
            # Un-normalize
            unnorm_returns = sampled_returns.squeeze().cpu().numpy() * data_std + data_mean
            
            # Rebuild prices
            prices = np.zeros_like(unnorm_returns)
            prices[:, 0] = 1.0 # S0
            for t in range(1, 128):
                prices[:, t] = prices[:, t-1] * np.exp(unnorm_returns[:, t])
                
            fig_price, ax_price = plt.subplots(figsize=(10, 4))
            for i in range(5):
                ax_price.plot(prices[i], alpha=0.8, label=f"Path {i+1}")
            ax_price.set_title("Synthetic Geometric Brownian Motion")
            ax_price.legend()
            st.pyplot(fig_price)
