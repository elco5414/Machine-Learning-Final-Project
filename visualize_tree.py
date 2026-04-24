"""
visualize_tree.py

Plot individual trees from the trained price model.
"""

import json
import os

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_tree

MODEL_DIR = "models/"


def load_model():
    """Load the saved model back into an XGBRegressor."""
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "price_model.json"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    # Attach feature names so the plot shows 'ret_5' instead of 'f2', etc.
    model.get_booster().feature_names = feature_cols
    return model


def plot_single_tree(model, tree_index=0, save_path=None):
    """Plot one tree from the ensemble."""
    n_trees = model.get_booster().num_boosted_rounds()
    if tree_index >= n_trees:
        raise ValueError(
            f"tree_index {tree_index} out of range (model has {n_trees} trees)"
        )

    fig, ax = plt.subplots(figsize=(30, 20))
    plot_tree(model, num_trees=tree_index, ax=ax)
    ax.set_title(f"Tree #{tree_index} of {n_trees}", fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    model = load_model()

    # Plot the very first tree (the "base" learner)
    plot_single_tree(model, tree_index=0, save_path="tree_0.png")

    # The last tree is interesting too — it's learning the residuals
    # that the previous 499 trees couldn't explain.
    n_trees = model.get_booster().num_boosted_rounds()
    plot_single_tree(model, tree_index=n_trees - 1, save_path=f"tree_{n_trees - 1}.png")
