import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time

class KMeansVisualizer:
    def __init__(self, X, K,small_delay = None):
        self.X = X
        self.K = K
        self.history = []  # To track centroid motion
        self.small_delay = small_delay if small_delay is not None else 0.1
        
        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(8.5, 6.5))
        plt.close(self.fig) # Prevent initial empty plot from showing up

    def update(self, curr_centroids, idx, iteration):
        """
        Call this inside your loop to update the plot live.
        """
        # 1. Store the position for the trace
        self.history.append(curr_centroids.copy())
        history_arr = np.array(self.history)
        
        # 2. Clear the current axes to redraw
        self.ax.clear()
        
        # 3. Plot Data Points (colored by current assignment)
        # Use a nice color map and some transparency
        scatter = self.ax.scatter(self.X[:, 0], self.X[:, 1], c=idx, cmap='viridis', 
                                  alpha=0.4, s=50, edgecolors='none')
        
        # 4. Trace the motion for each centroid with smooth lines
        # colors = ['#FF4444', '#4444FF', '#44BB44', '#FFAAAA', '#BB44BB']
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        for k in range(self.K):
            # Extract the path for centroid k
            path = history_arr[:, k, :]
            
            # Plot the line (trail) showing the history of movement
            color = colors[k % len(colors)]
            self.ax.plot(path[:, 0], path[:, 1], color=color, 
                         linestyle='-', marker='o', markersize=4, lw=1.5, alpha=0.8)
            
            # Plot current centroid position as a highlighted star
            self.ax.plot(curr_centroids[k, 0], curr_centroids[k, 1], marker='^', 
                         color=color, markersize=10, mec='black', mew=1.2, label=f'Centroid {k+1}')

        self.ax.set_title(f"K-Means Clustering - Iteration {iteration}", fontsize=14, pad=15)
        self.ax.set_xlabel("Feature 1", fontsize=12)
        self.ax.set_ylabel("Feature 2", fontsize=12)
        self.ax.grid(True, alpha=0.2)
        
        # Use IPython display for smooth "animation" in notebooks
        clear_output(wait=True)
        display(self.fig)
        time.sleep(self.small_delay) # Small delay for visual persistence

    def finish(self):
        # Final cleanup/display
        clear_output(wait=True)
        display(self.fig)
        plt.close(self.fig)