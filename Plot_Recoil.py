import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator


plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 3,  # Thicker edges for the plot
    'axes.titlesize': 35,         # Font size for the plot title
    'axes.labelsize': 37,         # Font size for x and y labels
    'axes.labelpad': 15,          # spacing between axis label and axis
    'legend.fontsize': 24,        # Font size for legend text
    'legend.title_fontsize': 26,  # Font size for legend title
    'xtick.direction': 'in',  # Ticks pointing inside
    'ytick.direction': 'in',  # Ticks pointing inside
    'xtick.major.size': 10,      # Length of major ticks
    'ytick.major.size': 10,      # Length of major ticks
    'xtick.major.width': 2,      # Thickness of major ticks
    'ytick.major.width': 2,      # Thickness of major ticks
    'xtick.minor.size': 5,       # Length of minor ticks
    'ytick.minor.size': 5,       # Length of minor ticks
    'xtick.minor.width': 1,      # Thickness of minor ticks
    'ytick.minor.width': 1,      # Thickness of minor ticks
    'xtick.labelsize': 28,       # Font size for tick labels
    'ytick.labelsize': 28,       # Font size for tick labels
})

RecoilGeNak = np.loadtxt("Nakazato_Recoil_Ge.dat")
RecoilSapNak = np.loadtxt("Nakazato_Recoil_Al2O3.dat")
RecoilGeLiv = np.loadtxt("Livermore_Recoil_Ge.dat")
RecoilSapLiv = np.loadtxt("Livermore_Recoil_Al2O3.dat")

# Create figure and main plot
fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)

# Plot the actual data
ax.plot(0.5*(RecoilSapNak[:,0]+ RecoilSapNak[:,1]),RecoilSapNak[:,2]/(RecoilSapNak[:,1]-RecoilSapNak[:,0]), linestyle="--", label="Al2O3,Nak",color='lime',linewidth=5)
ax.plot(0.5*(RecoilSapLiv[:,0]+ RecoilSapLiv[:,1]),RecoilSapLiv[:,2]/(RecoilSapLiv[:,1]-RecoilSapLiv[:,0]), linestyle="--", label="Al2O3,Liv",color='magenta',linewidth=5)
ax.plot(0.5*(RecoilGeNak[:,0]+ RecoilGeNak[:,1]),RecoilGeNak[:,2]/(RecoilGeNak[:,1]-RecoilGeNak[:,0]), linestyle="-", label="Ge,Nak",color='lime',linewidth=5)
ax.plot(0.5*(RecoilGeLiv[:,0]+ RecoilGeLiv[:,1]),RecoilGeLiv[:,2]/(RecoilGeLiv[:,1]-RecoilGeLiv[:,0]), linestyle="-", label="Ge,Liv",color='magenta',linewidth=5)

# Add gridlines (both major and minor)
ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)  # Major gridlines
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)  # Minor gridlines (for log scale)


# Move the table to the **top-right** but ensure it fits the log scale
inset_x, inset_y = 250, 50  # Adjust position based on log scale

# Define table cell positions
cell_x_start, cell_y_start = inset_x, inset_y
cell_x_offset, cell_y_offset = 70, 1.5  # Adjusted for better spacing in log scale

# Define line styles for each table entry
line_styles = {
    (0,0): ("lime", "-"),   # Ge, Nak
    (0,1): ("magenta", "-"),  # Ge, Liv
    (1,0): ("lime", "--"),     # Sap, Nak
    (1,1): ("magenta", "--")     # Sap, Liv
}

# Compute table y-coordinates for log scale
row_heights = [cell_y_start, cell_y_start / 10**0.5, cell_y_start / 10]  # Top and bottom y positions for rows


# Draw table grid
for i in range(3):
    ax.plot([cell_x_start, cell_x_start + 2 * cell_x_offset], 
            [cell_y_start / (10**(i * 0.5))] * 2, color='black', linewidth=2)

for j in range(3):
    ax.plot([cell_x_start + j * cell_x_offset] * 2, 
            [cell_y_start, cell_y_start / 10], color='black', linewidth=2)

# Labels for table columns & rows
ax.text(cell_x_start + cell_x_offset / 2, cell_y_start * 1.2, "Nak", fontsize=22, ha='center', fontweight='bold')
ax.text(cell_x_start + 3 * cell_x_offset / 2, cell_y_start * 1.2, "Liv", fontsize=22, ha='center', fontweight='bold')

# Move "Ge" and "Sap" slightly to the left to prevent overlap
ax.text(cell_x_start - 10, cell_y_start / 2, "Ge", fontsize=22, va='center', fontweight='bold', ha='right')
ax.text(cell_x_start - 10, cell_y_start / 5, r"$Al_2O_3$", fontsize=22, va='center', fontweight='bold', ha='right')

# Draw lines inside table cells
for (row, col), (color, style) in line_styles.items():
    x_pos = cell_x_start + (col + 0.5) * cell_x_offset
    y_top = row_heights[row]
    y_bottom = row_heights[row + 1] if row + 1 < len(row_heights) else row_heights[row] / 10
    y_pos = np.sqrt(y_top * y_bottom)  # Geometric mean for log scale positioning

    ax.plot([x_pos - 20, x_pos + 20], [y_pos, y_pos], color=color, linestyle=style, linewidth=5)


plt.xlabel("recoil energy [keV]")
plt.ylabel("events/keV [keV$^{-1}$]")

##plt.tight_layout()
plt.ylim(0.01,1e2)
plt.xlim(0,400)
plt.yscale("log")
plt.show()
