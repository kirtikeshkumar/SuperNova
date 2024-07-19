import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi / 4)
y4 = np.cos(x + np.pi / 4)
y5 = np.sin(x + np.pi / 2)
y6 = np.cos(x + np.pi / 2)

# Create a plot
fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='sin(x)')
line2, = ax.plot(x, y2, label='cos(x)')
line3, = ax.plot(x, y3, label='sin(x + π/4)')
line4, = ax.plot(x, y4, label='cos(x + π/4)')
line5, = ax.plot(x, y5, label='sin(x + π/2)')
line6, = ax.plot(x, y6, label='cos(x + π/2)')

# Custom legend entries (handles and labels)
handles = [line1, line2, line3, line4, line5, line6]
labels = ['sin(x)', 'cos(x)', 'sin(x + π/4)', 'cos(x + π/4)', 'sin(x + π/2)', 'cos(x + π/2)']

# Create the custom legend
leg = ax.legend(handles=handles[:6], labels=['', ''], ncol=1, numpoints=1, 
                borderaxespad=0., title='No prop.', framealpha=.75,
                facecolor='w', edgecolor='k', loc=2, fancybox=None)

# Add custom entries to the legend as a table
table_data = [
    ['Row 1, Col 1', 'Row 1, Col 2'],
    ['Row 2, Col 1', 'Row 2, Col 2'],
    ['Row 3, Col 1', 'Row 3, Col 2']
]

for i, row in enumerate(table_data):
    for j, text in enumerate(row):
        plt.text(0.5 + j * 0.15, 0.9 - i * 0.1, text, ha='center', va='center', fontsize=8,
                 transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot with Custom Legend as Table')

# Display the plot
plt.show()
