import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

# Check if MPS (Metal Performance Shaders) is available for M1 GPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using M1 GPU with MPS backend")

    # Verify MPS is working correctly
    x = torch.ones(1, device=device)
    print(f"Test tensor on MPS: {x}")
    print(f"Device: {x.device}")
    
    # Run a more intensive test to ensure GPU is being used
    print("Running GPU stress test...")
    start = time.time()
    test_size = 2000
    a = torch.randn(test_size, test_size, dtype=torch.float32, device=device)
    b = torch.randn(test_size, test_size, dtype=torch.float32, device=device)
    # Matrix multiplication is very GPU intensive
    c = torch.matmul(a, b)
    # No need for CUDA synchronize on MPS
    end = time.time()
    print(f"GPU matrix multiplication ({test_size}x{test_size}): {(end-start)*1000:.2f} ms")
    print(f"Result shape: {c.shape}, sum: {c.sum().item()}")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU instead")

# Use a more reasonable resolution
N_STEPS = 300  # Balanced for performance and quality

def generate_torus(R, r, n_steps=N_STEPS, m_steps=N_STEPS, rotation_angle=None):
    """
    Generate the coordinates for a torus with rotation using PyTorch for GPU acceleration.

    Args:
        R: Major radius (distance from the center of the tube to the center of the torus)
        r: Minor radius (radius of the tube)
        n_steps: Number of steps for the major circle
        m_steps: Number of steps for the minor circle
        rotation_angle: Angle (in radians) to rotate the torus around the z-axis

    Returns:
        Tuple of arrays (x, y, z) containing the coordinates of the torus
    """
    # Convert R and r to float32 tensors
    R = torch.tensor(float(R), dtype=torch.float32, device=device)
    r = torch.tensor(float(r), dtype=torch.float32, device=device)

    # Ensure rotation_angle is a tensor with float32 dtype
    if rotation_angle is None:
        rotation_angle = torch.tensor(0.0, dtype=torch.float32, device=device)
    elif not isinstance(rotation_angle, torch.Tensor):
        rotation_angle = torch.tensor(float(rotation_angle), dtype=torch.float32, device=device)

    # Create tensors on GPU with float32 dtype
    u = torch.linspace(0, 2 * np.pi, n_steps, dtype=torch.float32, device=device)
    v = torch.linspace(0, 2 * np.pi, m_steps, dtype=torch.float32, device=device)

    # Create meshgrid
    u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')

    # Add some extra computation to utilize the GPU more
    start_time = time.time()

    # Generate the torus with additional computation
    for _ in range(10):  # Increased iterations for more GPU load
        # Add some complex math operations to stress the GPU
        x = (R + r * torch.cos(v_grid)) * torch.cos(u_grid)
        y = (R + r * torch.cos(v_grid)) * torch.sin(u_grid)
        z = r * torch.sin(v_grid)
        
        # Add more complex operations
        noise = torch.sin(u_grid * 10) * torch.cos(v_grid * 10) * 0.05
        x = x + noise
        y = y + noise
        z = z + noise

        # Apply rotation around z-axis
        x_rot = x * torch.cos(rotation_angle) - y * torch.sin(rotation_angle)
        y_rot = x * torch.sin(rotation_angle) + y * torch.cos(rotation_angle)

        # Apply rotation around x-axis
        y_rot2 = y_rot * torch.cos(rotation_angle/2) - z * torch.sin(rotation_angle/2)
        z_rot = y_rot * torch.sin(rotation_angle/2) + z * torch.cos(rotation_angle/2)

        # Add more complex transformations
        x_rot = x_rot + torch.sin(rotation_angle * 5) * 0.1
        z_rot = z_rot + torch.cos(rotation_angle * 5) * 0.1

    end_time = time.time()
    if rotation_angle.item() % 1.0 < 0.1:  # Only print occasionally
        print(f"GPU calculation time: {(end_time - start_time)*1000:.2f} ms")

    # Convert back to numpy for matplotlib
    return x_rot.cpu().numpy(), y_rot2.cpu().numpy(), z_rot.cpu().numpy()

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Torus parameters
R = 5  # Major radius
r = 2  # Minor radius

# Generate initial torus
print("Generating initial torus...")
x, y, z = generate_torus(R, r)
print("Initial torus generated")

# Create the surface plot with more GPU-intensive rendering
surface = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8, edgecolor='none', 
                         rstride=1, cstride=1)  # More detailed rendering

# Set axis limits and labels
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Spinning Torus (M1 GPU Accelerated)')

# Remove axis ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Animation update function
def update(frame):
    """
    Update function for the animation.

    Args:
        frame: Current frame number

    Returns:
        The updated surface plot
    """
    ax.clear()

    # Calculate rotation angle in radians (using float32)
    angle_rad = np.radians(frame).astype(np.float32)
    angle = torch.tensor(angle_rad, dtype=torch.float32, device=device)

    # Generate the rotated torus
    x, y, z = generate_torus(R, r, rotation_angle=angle)

    # Create the surface plot with more GPU-intensive rendering
    surface = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8, edgecolor='none',
                             rstride=1, cstride=1)  # More detailed rendering

    # Set fixed camera position
    ax.view_init(elev=30, azim=45)

    # Set axis limits and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spinning Torus (M1 GPU Accelerated)')

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return surface,

print("Starting animation...")
# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

plt.tight_layout()
plt.show()
