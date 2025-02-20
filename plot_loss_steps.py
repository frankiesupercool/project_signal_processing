import pandas as pd
import matplotlib.pyplot as plt
import os
import config

plot_path = config.plot_folder + "loss_plot_steps.png"
log_dir = os.path.join(config.log_folder, "version_0", "metrics.csv")
df = pd.read_csv(log_dir)

train_loss_col = "train_loss_epoch"
val_loss_col = "val_loss"
train_loss_step_col = "train_loss_step"

plt.figure(figsize=(10, 6))

df_train_loss_step = df.dropna(subset=[train_loss_step_col])
plt.plot(df_train_loss_step['step'], df_train_loss_step[train_loss_step_col], label="Train Loss Step", color="#7f7f7f")


train_loss_epoch_valid = df.dropna(subset=[train_loss_col])
plt.plot(train_loss_epoch_valid['step'], train_loss_epoch_valid[train_loss_col], label="Train Loss Epoch", color='#1f77b4')
plt.scatter(train_loss_epoch_valid['step'], train_loss_epoch_valid[train_loss_col], color='#1f77b4', zorder=5, marker='o')

val_loss_valid = df.dropna(subset=[val_loss_col])
plt.plot(val_loss_valid['step'], val_loss_valid[val_loss_col], label="Validation Loss", color='#ff7f0e')
plt.scatter(val_loss_valid['step'], val_loss_valid[val_loss_col], color='#ff7f0e', zorder=5, marker='o')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Losses over Steps')
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
