import os
import matplotlib.pyplot as plt
import pandas as pd
import config

"""
Plots train / val loss from lightning logs
"""

plot_path = os.path.join(config.plot_folder, "loss_plot_epochs.png")
log_dir = os.path.join(config.log_folder, "lightning_logs")
log_dir = os.path.join(log_dir, "version_0", "metrics.csv")
df = pd.read_csv(log_dir)

# print(df.head())

train_loss_col = "train_loss_epoch"
val_loss_col = "val_loss"

plt.figure(figsize=(10, 5))

# Ignore NaN values, log includes also train loss step (after each 100 steps), line then includes NaN for train
# and val loss until end of epoch
if train_loss_col:
    df_train = df[["epoch", train_loss_col]].dropna()
    plt.plot(df_train["epoch"], df_train[train_loss_col], label="Train Loss")

if val_loss_col:
    df_val = df[["epoch", val_loss_col]].dropna()
    plt.plot(df_val["epoch"], df_val[val_loss_col], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
