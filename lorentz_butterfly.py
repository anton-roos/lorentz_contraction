import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Lorenz parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz equations
def lorenz_equations(state, t):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time points
t = np.linspace(0, 50, 5000)
dt = t[1] - t[0]

# Initial conditions
x0 = [1.0, 1.0, 1.0]

# Solve the differential equations
def solve_lorenz(x0, t):
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))
    
    x[0], y[0], z[0] = x0
    
    for i in range(1, len(t)):
        dx_dt, dy_dt, dz_dt = lorenz_equations([x[i-1], y[i-1], z[i-1]], t[i-1])
        x[i] = x[i-1] + dx_dt * dt
        y[i] = y[i-1] + dy_dt * dt
        z[i] = z[i-1] + dz_dt * dt
    
    return x, y, z

# Calculate the trajectory
x, y, z = solve_lorenz(x0, t)

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the line object
line, = ax.plot([], [], [], 'b-', alpha=0.7)
point, = ax.plot([], [], [], 'ro')

# Set axis limits
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor - Butterfly Effect')

# Initialization function
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

# Animation update function
def update(frame):
    # Update line (full trajectory up to frame)
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    
    # Update point (current position)
    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])
    
    return line, point

# Create animation
ani = FuncAnimation(fig, update, frames=range(0, len(t), 10),
                   init_func=init, blit=True, interval=20)

plt.show()
