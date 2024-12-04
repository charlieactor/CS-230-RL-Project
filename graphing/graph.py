import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    # Data for Old Search Algorithm
    # Taken from scenario runner outputs over 10 runs
    old_search_moves = [672, 776, 692.0, 750.8]
    old_search_z_moves = [50, 76, 56.0, 75.1]
    old_search_picks_drops = [190, 228, 202.0, 222.6]

    # Data for RL Model
    # Taken from scenario runner outputs over 10 runs
    rl_moves = [443, 527, 476.0, 521.15]
    rl_z_moves = [25, 54, 32.0, 53.55]
    rl_picks_drops = [102, 150, 121.0, 140.1]

    # Labels for the statistics
    labels = ["Minimum", "Maximum", "P50", "P95"]

    # Plotting setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Graph 1: Bot Moves
    axes[0].bar(labels, old_search_moves, width=0.4, label="Old Search Algo", align="center", alpha=0.7)
    axes[0].bar(labels, rl_moves, width=0.4, label="RL Model", align="edge", alpha=0.7)
    axes[0].set_title("Bot Moves")
    axes[0].set_ylabel("Move Count")
    axes[0].legend()

    # Graph 2: Bot Z Moves
    axes[1].bar(labels, old_search_z_moves, width=0.4, label="Old Search Algo", align="center", alpha=0.7)
    axes[1].bar(labels, rl_z_moves, width=0.4, label="RL Model", align="edge", alpha=0.7)
    axes[1].set_title("Bot Z Moves")
    axes[1].set_ylabel("Move Count")
    axes[1].legend()

    # Graph 3: Bot Picks/Drops
    axes[2].bar(labels, old_search_picks_drops, width=0.4, label="Old Search Algo", align="center", alpha=0.7)
    axes[2].bar(labels, rl_picks_drops, width=0.4, label="RL Model", align="edge", alpha=0.7)
    axes[2].set_title("Bot Picks/Drops")
    axes[2].set_ylabel("Picks/Drops Count")
    axes[2].legend()

    # Final layout adjustments
    plt.tight_layout()
    plt.show()
