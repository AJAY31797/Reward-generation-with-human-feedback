import numpy as np
import matplotlib.pyplot as plt

path = # Add your path

iterations = [1, 2, 3, 4]
window = 5

def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode="valid")

for i in iterations:
    reward_margins_path = # Add your path
    reward_margins = np.load(reward_margins_path)
    length = len(reward_margins)
    x = list(range(1, length + 1))
    y = reward_margins

    # Moving average
    y_ma = moving_average(y, window)
    x_ma = x[window - 1:]  # align with MA output

    fig, ax = plt.subplots(figsize=(6, 4), dpi = 300) # Creates a figure and a set of subplots 
    
    # Raw curve
    ax.plot(x, y, linewidth=1.2, alpha=0.6, label="Raw")

    # Moving average curve
    ax.plot(
        x_ma,
        y_ma,
        linewidth=1.2,
        linestyle="--",
        color="orange",
        label=f"{window}-epoch moving average"
    )
         # Plots the line on the axes

    # Labels and title with thin font weight
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Reward Margin', fontsize=12)
    ax.set_title(
        f'Reward Margin over Training Epochs for Iteration {i}',
        fontsize=14
    )

    # Grid with thin lines
    ax.grid(True, linewidth=0.5)

    # Thin axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Thin ticks
    ax.tick_params(width=0.5, labelsize=12)

    # Legend (thin and clean)
    ax.legend(frameon=False, fontsize=12)

    # Save at 300 DPI
    plt.savefig(
        path + f"reward_margin_plot_{i}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
