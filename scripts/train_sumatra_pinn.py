#!/usr/bin/env python3
"""
Train PINN on 2004 Sumatra-Andaman Earthquake Data

This script:
1. Downloads real seismic waveforms from IRIS
2. Generates a regional velocity model
3. Trains a PINN on the earthquake data
4. Creates visualizations of waveforms and predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
output_dir = Path("outputs/sumatra_2004")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 1: Load Real Seismic Data
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Loading 2004 Sumatra-Andaman Earthquake Data")
print("="*70)

from src.data.seismic_data_loader import (
    IRISDataLoader,
    SumatraVelocityModel,
    MAJOR_EVENTS
)

# Get event info
event = MAJOR_EVENTS["sumatra_2004"]
print(f"\nEvent: {event.name}")
print(f"Magnitude: {event.magnitude} Mw")
print(f"Location: {event.latitude}°N, {event.longitude}°E")
print(f"Depth: {event.depth_km} km")
print(f"Date: {event.origin_time}")
print(f"Rupture length: {event.rupture_length_km} km")

# Download waveforms
loader = IRISDataLoader()
print("\nDownloading seismic waveforms from IRIS...")
waveforms = loader.download_event_data(
    "sumatra_2004",
    duration_minutes=30,
    channels="BHZ"
)
print(f"Downloaded {len(waveforms)} waveforms from {len(set(w.station.station for w in waveforms))} stations")

# Preprocess waveforms
print("\nPreprocessing waveforms...")
processed = loader.preprocess_waveforms(
    waveforms,
    filter_band=(0.01, 0.5),  # Focus on long-period waves for tsunami
)
print(f"Preprocessed {len(processed)} waveforms")

# ============================================================================
# Step 2: Visualize Raw Waveforms
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Visualizing Seismic Waveforms")
print("="*70)

# Sort by distance from epicenter
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Calculate distances and sort
waveform_distances = []
for wf in processed:
    dist = haversine_distance(
        event.latitude, event.longitude,
        wf.station.latitude, wf.station.longitude
    )
    waveform_distances.append((wf, dist))

waveform_distances.sort(key=lambda x: x[1])

# Plot waveforms sorted by distance
fig, axes = plt.subplots(min(12, len(processed)), 1, figsize=(14, 16))
if len(processed) == 1:
    axes = [axes]

for idx, (wf, dist) in enumerate(waveform_distances[:12]):
    ax = axes[idx]

    # Time axis
    t = np.arange(len(wf.data)) / wf.sampling_rate

    # Normalize for display
    data_norm = wf.data / (np.max(np.abs(wf.data)) + 1e-10)

    ax.plot(t, data_norm, 'b-', linewidth=0.5)
    ax.set_ylabel(f"{wf.station.station}\n{dist:.0f} km", fontsize=8)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.set_title(f"2004 Sumatra-Andaman Earthquake (M{event.magnitude}) - Seismic Waveforms\n"
                    f"Sorted by distance from epicenter", fontsize=12)
    if idx == len(axes) - 1:
        ax.set_xlabel("Time after origin (seconds)")

plt.tight_layout()
plt.savefig(output_dir / "sumatra_waveforms.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_waveforms.png'}")
plt.close()

# ============================================================================
# Step 3: Create Station Map
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Creating Station Map")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot stations
unique_stations = {}
for wf in processed:
    key = wf.station.station
    if key not in unique_stations:
        unique_stations[key] = wf.station

lats = [s.latitude for s in unique_stations.values()]
lons = [s.longitude for s in unique_stations.values()]
names = list(unique_stations.keys())

# Plot epicenter
ax.scatter([event.longitude], [event.latitude], c='red', s=500, marker='*',
           edgecolors='black', linewidths=1, zorder=10, label='Epicenter')

# Plot stations
scatter = ax.scatter(lons, lats, c='blue', s=100, marker='^',
                     edgecolors='black', linewidths=0.5, zorder=5, label='Seismic Stations')

# Add station labels
for name, lon, lat in zip(names, lons, lats):
    ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=8)

# Draw lines from epicenter to stations
for lon, lat in zip(lons, lats):
    ax.plot([event.longitude, lon], [event.latitude, lat],
            'b-', alpha=0.2, linewidth=0.5)

# Add rupture zone (approximate)
rupture_lat = np.linspace(3.3, 14, 50)
rupture_lon = np.linspace(95.9, 93, 50)
ax.plot(rupture_lon, rupture_lat, 'r-', linewidth=3, alpha=0.7, label='Rupture Zone (~1300 km)')

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"2004 Sumatra-Andaman Earthquake - Global Seismic Network\n"
            f"M{event.magnitude} | {event.origin_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 80)

plt.tight_layout()
plt.savefig(output_dir / "sumatra_station_map.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_station_map.png'}")
plt.close()

# ============================================================================
# Step 4: Generate Velocity Model
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Generating Sumatra Velocity Model")
print("="*70)

velocity_model = SumatraVelocityModel(
    nx=200,
    nz=100,
    x_extent_km=500,
    z_extent_km=200
)

velocity_data = velocity_model.to_pinn_format()
velocity = np.array(velocity_data["velocity"])

print(f"Velocity model shape: {velocity.shape}")
print(f"Velocity range: {velocity.min():.2f} - {velocity.max():.2f} km/s")

# Plot velocity model
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

x = np.linspace(0, velocity_data["x_extent_km"], velocity_data["shape"][1])
z = np.linspace(0, velocity_data["z_extent_km"], velocity_data["shape"][0])
X, Z = np.meshgrid(x, z)

im = ax.pcolormesh(X, Z, velocity, shading='auto', cmap='seismic')
ax.invert_yaxis()
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Depth (km)")
ax.set_title(f"Sumatra-Andaman Subduction Zone - P-wave Velocity Model\n"
            f"Including oceanic crust, accretionary wedge, and subducting slab")

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label="P-wave Velocity (km/s)")

# Mark approximate features
ax.axhline(y=30, color='white', linestyle='--', linewidth=1, alpha=0.7, label='Moho')
ax.text(10, 25, 'Continental Crust', color='white', fontsize=10)
ax.text(10, 80, 'Upper Mantle', color='white', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "sumatra_velocity_model.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_velocity_model.png'}")
plt.close()

# ============================================================================
# Step 5: Prepare Training Data
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Preparing PINN Training Data")
print("="*70)

# Convert to training format
training_data = loader.to_training_data(
    processed,
    event,
    downsample_factor=20,  # Reduce for faster training
)

coords = training_data["coordinates"]  # [x, z, t]
obs = training_data["observations"]     # Amplitudes

print(f"Training samples: {coords.shape[0]}")
print(f"Coordinate dimensions: {coords.shape[1]} (x, z, t)")
print(f"Observation range: [{obs.min():.4f}, {obs.max():.4f}]")

# Convert to tensors
coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
if obs_tensor.dim() == 1:
    obs_tensor = obs_tensor.unsqueeze(1)

# Normalize coordinates
coord_min = coords_tensor.min(dim=0)[0]
coord_max = coords_tensor.max(dim=0)[0]
coords_norm = (coords_tensor - coord_min) / (coord_max - coord_min + 1e-8)

print(f"Normalized coordinate ranges: [0, 1]")

# ============================================================================
# Step 6: Train PINN
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Training Physics-Informed Neural Network")
print("="*70)

from src.models import SIRENNetwork

# Create PINN model
model = SIRENNetwork(
    in_features=3,       # x, z, t
    out_features=1,      # wavefield amplitude
    hidden_features=128,
    hidden_layers=4,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

# Training loop
n_epochs = 500
batch_size = 4096
n_samples = coords_norm.shape[0]

losses = []
physics_losses = []

print(f"\nTraining for {n_epochs} epochs...")
print("-" * 50)

for epoch in range(n_epochs):
    model.train()

    # Random batch
    idx = torch.randperm(n_samples)[:batch_size]
    batch_coords = coords_norm[idx]
    batch_obs = obs_tensor[idx]

    # Enable gradients for physics loss
    batch_coords.requires_grad_(True)

    # Forward pass
    pred = model(batch_coords)

    # Data loss
    data_loss = torch.nn.functional.mse_loss(pred, batch_obs)

    # Physics loss (wave equation residual)
    # ∂²u/∂t² = c² * (∂²u/∂x² + ∂²u/∂z²)
    grads = torch.autograd.grad(
        pred, batch_coords,
        grad_outputs=torch.ones_like(pred),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_z = grads[:, 1:2]
    u_t = grads[:, 2:3]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, batch_coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_zz = torch.autograd.grad(u_z, batch_coords, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 1:2]
    u_tt = torch.autograd.grad(u_t, batch_coords, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 2:3]

    # Approximate velocity (normalized)
    c = 0.5  # Normalized velocity
    physics_residual = u_tt - c**2 * (u_xx + u_zz)
    physics_loss = torch.mean(physics_residual**2)

    # Total loss
    total_loss = data_loss + 0.1 * physics_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(total_loss.item())
    physics_losses.append(physics_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {total_loss.item():.6f} | "
              f"Data: {data_loss.item():.6f} | Physics: {physics_loss.item():.6f}")

print("-" * 50)
print("Training complete!")

# ============================================================================
# Step 7: Visualize Training Progress
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Visualizing Training Progress")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax = axes[0]
ax.semilogy(losses, 'b-', label='Total Loss', alpha=0.8)
ax.semilogy(physics_losses, 'r-', label='Physics Loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Prediction vs actual for sample waveform
ax = axes[1]
model.eval()
with torch.no_grad():
    # Get predictions for first station
    first_station_mask = (coords[:, 0] == coords[0, 0]) & (coords[:, 1] == coords[0, 1])
    station_coords = coords_norm[first_station_mask]
    station_obs = obs_tensor[first_station_mask]
    station_pred = model(station_coords)

    t_station = coords[first_station_mask, 2]
    sort_idx = np.argsort(t_station)

    ax.plot(t_station[sort_idx], station_obs.cpu().numpy().flatten()[sort_idx],
            'b-', label='Observed', alpha=0.7)
    ax.plot(t_station[sort_idx], station_pred.cpu().numpy().flatten()[sort_idx],
            'r--', label='PINN Prediction', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Waveform Fit - Sample Station')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "sumatra_training_results.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_training_results.png'}")
plt.close()

# ============================================================================
# Step 8: Generate Wavefield Snapshots
# ============================================================================
print("\n" + "="*70)
print("STEP 8: Generating Wavefield Snapshots")
print("="*70)

# Create grid for visualization
nx_viz, nz_viz = 100, 50
x_viz = np.linspace(0, 1, nx_viz)
z_viz = np.linspace(0, 1, nz_viz)
X_viz, Z_viz = np.meshgrid(x_viz, z_viz)

# Generate snapshots at different times
times = [0.2, 0.4, 0.6, 0.8]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

model.eval()
with torch.no_grad():
    for idx, t in enumerate(times):
        ax = axes[idx // 2, idx % 2]

        # Create coordinates for this time
        coords_viz = np.stack([
            X_viz.flatten(),
            Z_viz.flatten(),
            np.full(nx_viz * nz_viz, t)
        ], axis=1)

        coords_viz_tensor = torch.tensor(coords_viz, dtype=torch.float32, device=device)
        wavefield = model(coords_viz_tensor).cpu().numpy().reshape(nz_viz, nx_viz)

        im = ax.pcolormesh(X_viz, Z_viz, wavefield, shading='auto', cmap='seismic',
                          vmin=-0.5, vmax=0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Depth')
        ax.set_title(f't = {t:.1f} (normalized)')
        plt.colorbar(im, ax=ax, label='Amplitude')

fig.suptitle('PINN Predicted Wavefield Evolution - 2004 Sumatra Earthquake', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "sumatra_wavefield_snapshots.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_wavefield_snapshots.png'}")
plt.close()

# ============================================================================
# Step 9: Summary Visualization
# ============================================================================
print("\n" + "="*70)
print("STEP 9: Creating Summary Visualization")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Station map (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter([event.longitude], [event.latitude], c='red', s=200, marker='*',
           edgecolors='black', linewidths=1, zorder=10)
ax1.scatter(lons, lats, c='blue', s=50, marker='^', edgecolors='black', linewidths=0.5)
ax1.set_xlim(40, 180)
ax1.set_ylim(-50, 60)
ax1.set_title('Seismic Network', fontsize=10)
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)

# 2. Velocity model (top middle and right)
ax2 = fig.add_subplot(gs[0, 1:])
# Create grid matching velocity model dimensions
x_vel = np.linspace(0, velocity_data["x_extent_km"], velocity.shape[1])
z_vel = np.linspace(0, velocity_data["z_extent_km"], velocity.shape[0])
X_vel, Z_vel = np.meshgrid(x_vel, z_vel)
im2 = ax2.pcolormesh(X_vel, Z_vel, velocity, shading='auto', cmap='seismic')
ax2.invert_yaxis()
ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Depth (km)')
ax2.set_title('Sumatra Subduction Zone Velocity Model', fontsize=10)
plt.colorbar(im2, ax=ax2, label='Vp (km/s)')

# 3. Waveforms (middle row)
ax3 = fig.add_subplot(gs[1, :])
for idx, (wf, dist) in enumerate(waveform_distances[:8]):
    t = np.arange(len(wf.data)) / wf.sampling_rate
    data_norm = wf.data / (np.max(np.abs(wf.data)) + 1e-10) * 0.4 + idx
    ax3.plot(t, data_norm, 'b-', linewidth=0.5)
    ax3.text(-50, idx, f"{wf.station.station} ({dist:.0f} km)", fontsize=8, ha='right')
ax3.set_xlim(-60, t[-1])
ax3.set_xlabel('Time (s)')
ax3.set_title('Recorded Seismic Waveforms (sorted by distance)', fontsize=10)
ax3.set_yticks([])

# 4. Training loss (bottom left)
ax4 = fig.add_subplot(gs[2, 0])
ax4.semilogy(losses, 'b-', label='Total', alpha=0.8)
ax4.semilogy(physics_losses, 'r-', label='Physics', alpha=0.8)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Training Progress', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. PINN prediction snapshots (bottom middle and right)
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[2, 2])

with torch.no_grad():
    for ax, t_val in [(ax5, 0.3), (ax6, 0.7)]:
        coords_viz = np.stack([
            X_viz.flatten(),
            Z_viz.flatten(),
            np.full(nx_viz * nz_viz, t_val)
        ], axis=1)
        coords_viz_tensor = torch.tensor(coords_viz, dtype=torch.float32, device=device)
        wavefield = model(coords_viz_tensor).cpu().numpy().reshape(nz_viz, nx_viz)

        im = ax.pcolormesh(X_viz, Z_viz, wavefield, shading='auto', cmap='seismic',
                          vmin=-0.5, vmax=0.5)
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Depth')
        ax.set_title(f'PINN Wavefield (t={t_val})', fontsize=10)

fig.suptitle(f'2004 Sumatra-Andaman Earthquake (M{event.magnitude}) - PINN Analysis\n'
            f'{event.origin_time.strftime("%Y-%m-%d %H:%M:%S UTC")}',
            fontsize=14, fontweight='bold')

plt.savefig(output_dir / "sumatra_summary.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'sumatra_summary.png'}")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"""
2004 Sumatra-Andaman Earthquake PINN Analysis

Event Details:
  - Magnitude: {event.magnitude} Mw
  - Location: {event.latitude}°N, {event.longitude}°E
  - Depth: {event.depth_km} km
  - Rupture: {event.rupture_length_km} km long

Data:
  - Waveforms downloaded: {len(waveforms)}
  - Stations used: {len(unique_stations)}
  - Training samples: {coords.shape[0]}

Model:
  - Architecture: SIREN (4 hidden layers, 128 units)
  - Parameters: {sum(p.numel() for p in model.parameters()):,}
  - Final loss: {losses[-1]:.6f}

Output files saved to: {output_dir}/
  - sumatra_waveforms.png
  - sumatra_station_map.png
  - sumatra_velocity_model.png
  - sumatra_training_results.png
  - sumatra_wavefield_snapshots.png
  - sumatra_summary.png
""")

# Save model checkpoint
checkpoint_path = output_dir / "sumatra_pinn_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
    'physics_losses': physics_losses,
    'event': event.name,
    'n_stations': len(unique_stations),
    'n_samples': coords.shape[0],
}, checkpoint_path)
print(f"Model saved to: {checkpoint_path}")
