import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

size = 100
steps = 60
p_tree = 0.65
p_ignite = 0.0005

grid = np.random.choice([0, 1], size=(size, size), p=[1 - p_tree, p_tree])
grid[size // 2, size // 2] = 2

initial_trees = np.sum(grid == 1) + 1
print(f"Grid Size: {size}x{size} | Initial Trees: {initial_trees}")
print("-" * 55)

for step in range(1, steps + 1):
    burning = (grid == 2)
    neighbors_burning = (
        np.roll(burning, 1, axis=0) |
        np.roll(burning, -1, axis=0) |
        np.roll(burning, 1, axis=1) |
        np.roll(burning, -1, axis=1)
    )
    neighbors_burning[0, :] = False
    neighbors_burning[-1, :] = False
    neighbors_burning[:, 0] = False
    neighbors_burning[:, -1] = False

    new_grid = grid.copy()
    new_grid[burning] = 3
    ignite = (grid == 1) & (neighbors_burning | (np.random.rand(size, size) < p_ignite))
    new_grid[ignite] = 2
    grid = new_grid

    if step % 15 == 0 or step == 1:
        trees = np.sum(grid == 1)
        active = np.sum(grid == 2)
        burned = np.sum(grid == 3)
        print(f"Step {step:2d} | Trees: {trees:4d} | Burning: {active:3d} | Burned: {burned:4d}")

final_trees = np.sum(grid == 1)
final_burned = np.sum(grid == 3) + np.sum(grid == 2)
burn_pct = (final_burned / initial_trees) * 100

print("-" * 55)
print(f"Remaining Trees : {final_trees}")
print(f"Total Burned    : {final_burned} ({burn_pct:.1f}% of forest)")

cmap = ListedColormap(['#1e1e1e', '#2e8b57', '#ff4500', '#696969'])
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
cbar = plt.colorbar(ticks=[0, 1, 2, 3], shrink=0.8)
cbar.ax.set_yticklabels(['Empty', 'Tree', 'Burning', 'Burned'])
plt.title('Parallel Cellular Forest Fire Spread')
plt.axis('off')
plt.tight_layout()
plt.show()
