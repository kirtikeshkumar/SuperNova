import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

## Plot for table like legend
### Generate sample data
##x = np.linspace(0, 500, 100)
##y1 = x / 50
##y2 = (x / 500) ** 0.5
##y3 = np.log1p(x) / 5
##y4 = np.exp(x / 1000)
##
### Create figure and main plot
##fig, ax = plt.subplots(figsize=(8,6))
##
### Plot the actual data
##ax.plot(x, y1, color='orange', linestyle='-', label="Ge, Nak")
##ax.plot(x, y2, color='blue', linestyle='-', label="Sap, Nak")
##ax.plot(x, y3, color='orange', linestyle='--', label="Ge, Liv")
##ax.plot(x, y4, color='blue', linestyle='--', label="Sap, Liv")
##
### Set log scale for y-axis
##ax.set_yscale("log")
##
### Set x and y limits
##ax.set_xlim(0, 500)
##ax.set_ylim(0.01, 100)
##
### Move the table to the **top-right** but ensure it fits the log scale
##inset_x, inset_y = 350, 50  # Adjust position based on log scale
##
### Define table cell positions
##cell_x_start, cell_y_start = inset_x, inset_y
##cell_x_offset, cell_y_offset = 70, 1.5  # Adjusted for better spacing in log scale
##
### Define line styles for each table entry
##line_styles = {
##    (0,0): ("orange", "-"),   # Ge, Nak
##    (0,1): ("orange", "--"),  # Ge, Liv
##    (1,0): ("blue", "-"),     # Sap, Nak
##    (1,1): ("blue", "--")     # Sap, Liv
##}
##
### Compute table y-coordinates for log scale
##row_heights = [cell_y_start, cell_y_start / 10**0.5, cell_y_start / 10]  # Top and bottom y positions for rows
##
##
### Draw table grid
##for i in range(3):
##    ax.plot([cell_x_start, cell_x_start + 2 * cell_x_offset], 
##            [cell_y_start / (10**(i * 0.5))] * 2, color='black', linewidth=1)
##
##for j in range(3):
##    ax.plot([cell_x_start + j * cell_x_offset] * 2, 
##            [cell_y_start, cell_y_start / 10], color='black', linewidth=1)
##
### Labels for table columns & rows
##ax.text(cell_x_start + cell_x_offset / 2, cell_y_start * 1.2, "Nak", fontsize=12, ha='center', fontweight='bold')
##ax.text(cell_x_start + 3 * cell_x_offset / 2, cell_y_start * 1.2, "Liv", fontsize=12, ha='center', fontweight='bold')
##
### Move "Ge" and "Sap" slightly to the left to prevent overlap
##ax.text(cell_x_start - 10, cell_y_start / 2, "Ge", fontsize=12, va='center', fontweight='bold', ha='right')
##ax.text(cell_x_start - 10, cell_y_start / 5, "Sap", fontsize=12, va='center', fontweight='bold', ha='right')
##
### Draw lines inside table cells
##for (row, col), (color, style) in line_styles.items():
##    x_pos = cell_x_start + (col + 0.5) * cell_x_offset
##    y_top = row_heights[row]
##    y_bottom = row_heights[row + 1] if row + 1 < len(row_heights) else row_heights[row] / 10
##    y_pos = np.sqrt(y_top * y_bottom)  # Geometric mean for log scale positioning
##
##    ax.plot([x_pos - 20, x_pos + 20], [y_pos, y_pos], color=color, linestyle=style, linewidth=2)
##
### Show the plot
##plt.show()
##
##import matplotlib.pyplot as plt
##import numpy as np
##
### Generate sample data
##x = np.linspace(0, 500, 100)
##y1 = x / 250 * 2.1e56
##y2 = (x / 500) * 2.1e56
##y3 = np.log1p(x) / 2.1 * 2.1e56
##y4 = np.exp(x / 1000) / 10 * 2.1e56
##y5 = np.sin(x / 500 * np.pi) * 2.1e56
##y6 = np.cos(x / 500 * np.pi) * 2.1e56
##
### Create figure and main plot
##fig, ax = plt.subplots(figsize=(8,6))
##
### Plot the actual data
##ax.plot(x, y1, color='orange', linestyle='-', label="Ge, Nak")
##ax.plot(x, y2, color='blue', linestyle='-', label="Sap, Nak")
##ax.plot(x, y3, color='orange', linestyle='--', label="Ge, Liv")
##ax.plot(x, y4, color='blue', linestyle='--', label="Sap, Liv")
##ax.plot(x, y5, color='green', linestyle='-.', label="Ge, Alt")
##ax.plot(x, y6, color='red', linestyle='-.', label="Sap, Alt")
##
### Set linear scale for y-axis
##ax.set_xscale("linear")
##ax.set_yscale("linear")
##
### Set x and y limits
##ax.set_xlim(0, 500)
##ax.set_ylim(0, 2.1e56)
##
### Add gridlines
##ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
##
### Table position in plot
##inset_x, inset_y = 350, 1.8e56  # Adjusted position
##
### Define table cell positions
##cell_x_start, cell_y_start = inset_x, inset_y
##cell_x_offset, cell_y_offset = 70, 0.6e56  # Adjust for large values
##
### Define line styles for each table entry
##line_styles = {
##    (0,0): ("orange", "-"),   # Ge, Nak
##    (0,1): ("orange", "--"),  # Ge, Liv
##    (1,0): ("blue", "-"),     # Sap, Nak
##    (1,1): ("blue", "--"),    # Sap, Liv
##    (2,0): ("green", "-."),   # Ge, Alt
##    (2,1): ("red", "-.")      # Sap, Alt
##}
##
### Compute table y-coordinates
##row_heights = [cell_y_start - i * cell_y_offset for i in range(4)]
##
### Draw table grid
##for i in range(4):
##    ax.plot([cell_x_start, cell_x_start + 2 * cell_x_offset], 
##            [row_heights[i]] * 2, color='black', linewidth=1)
##
##for j in range(3):
##    ax.plot([cell_x_start + j * cell_x_offset] * 2, 
##            [row_heights[0], row_heights[-1]], color='black', linewidth=1)
##
### Labels for table columns & rows
##ax.text(cell_x_start + cell_x_offset / 2, row_heights[0] + 0.2e56, "Nak", fontsize=12, ha='center', fontweight='bold')
##ax.text(cell_x_start + 3 * cell_x_offset / 2, row_heights[0] + 0.2e56, "Liv", fontsize=12, ha='center', fontweight='bold')
##
### Move "Ge", "Sap", "Alt" slightly to the left
##labels = ["Ge", "Sap", "Alt"]
##for i, label in enumerate(labels):
##    ax.text(cell_x_start - 10, (row_heights[i] + row_heights[i+1]) / 2, label, 
##            fontsize=12, va='center', fontweight='bold', ha='right')
##
### Draw lines inside table cells
##for (row, col), (color, style) in line_styles.items():
##    x_pos = cell_x_start + (col + 0.5) * cell_x_offset
##    y_pos = (row_heights[row] + row_heights[row+1]) / 2  # Midpoint for each row
##
##    ax.plot([x_pos - 20, x_pos + 20], [y_pos, y_pos], color=color, linestyle=style, linewidth=2)
##
### Show the plot
##plt.show()


## Inset Plot

# Sample data (replace with your actual data)
x = np.linspace(0, 1000, 500)  # 500 data points
y = np.sin(x / 50) * np.exp(-x / 1000)  # Some function for y

# Create figure and main plot
fig, ax_main = plt.subplots(figsize=(8,6))

fig, ax_main = plt.subplots(figsize=(8,6))
ax_main.plot(x, y, label="Full Data", color="blue")
ax_main.set_yscale("log")
ax_main.legend(loc="upper right")
ax_main.grid(True)

# Create inset plot (bottom-left)
ax_inset = fig.add_axes([0.15, 0.15, 0.3, 0.3])
ax_inset.plot(x[:100], y[:100], color="red", label="First 100 points")
ax_inset.set_yscale("linear")
ax_inset.grid(True)

# 🔹 Set custom yticks and scientific notation
ax_inset.set_yticks([100, 200, 300, 400, 500, 600])  # Only show 1, 2, 3, 4, 5, 6 (×10²)
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 2))  # Force 10² notation
ax_inset.yaxis.set_major_formatter(formatter)

# 🔹 Adjust font size of the scale factor (10²)
ax_inset.yaxis.get_offset_text().set_fontsize(12)

plt.show()

