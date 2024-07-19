import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data (replace with your actual data)
x_positions = np.linspace(-5, 5, 100)
y_positions = np.linspace(1, 100, 100)  # Example with a large range of y-values
x_grid, y_grid = np.meshgrid(x_positions, y_positions)
grid_values = np.sin(np.sqrt(x_grid**2 + y_grid**2))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create surface plot
surf = ax.plot_surface(x_grid, y_grid, grid_values, cmap='viridis')

# Apply logarithmic scale on y-axis
ax.set_yscale('log')

# Add color bar which maps values to colors
fig.colorbar(surf)

# Set labels
ax.set_xlabel('X Positions')
ax.set_ylabel('Y Positions (Log Scale)')
ax.set_zlabel('Values')

plt.title('Surface Plot with Logarithmic Y-axis')
plt.show()
