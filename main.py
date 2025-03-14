import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
c = 299792458  # Speed of light in m/s
L0 = 1.0       # Rest length of object (1 meter)

# Function to calculate Lorentz contraction
def lorentz_contraction(v):
    """Calculate contracted length based on velocity"""
    gamma = 1 / np.sqrt(1 - (v**2 / c**2))
    return L0 / gamma

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Length (m)')
ax.set_ylabel('Y')
ax.set_title('Lorentz Contraction Simulation')
ax.grid(True)

# Create object (rectangle) to animate
rect = plt.Rectangle((-L0/2, -0.1), L0, 0.2, fc='blue', alpha=0.5)
ax.add_patch(rect)

# Text to display velocity and contracted length
velocity_text = ax.text(-1.4, 0.3, '', fontsize=12)
length_text = ax.text(-1.4, 0.25, '', fontsize=12)

# Animation initialization
def init():
    rect.set_width(L0)
    velocity_text.set_text(f'Velocity: 0 m/s')
    length_text.set_text(f'Contracted Length: {L0:.3f} m')
    return rect, velocity_text, length_text

# Animation update function
def update(frame):
    # Calculate velocity (fraction of speed of light)
    v = frame * c / 100  # Frame goes from 0 to 99
    
    # Calculate contracted length
    L = lorentz_contraction(v)
    
    # Update rectangle
    rect.set_width(L)
    rect.set_x(-L/2)  # Keep centered
    
    # Update text
    velocity_text.set_text(f'Velocity: {v:.2e} m/s ({frame/100:.2f}c)')
    length_text.set_text(f'Contracted Length: {L:.3f} m')
    
    return rect, velocity_text, length_text

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), 
                   init_func=init, blit=True, interval=50)

plt.show()
