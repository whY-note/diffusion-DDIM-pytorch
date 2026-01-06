import pandas as pd
import matplotlib.pyplot as plt

dataset_name = "butterflies"
loss_filename = dataset_name + "_loss.csv"
df = pd.read_csv("./"+ loss_filename)

plt.plot(df["epoch"], df["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()

plt.savefig(dataset_name+"_loss.png", dpi=300, bbox_inches="tight")


##########################################

# import pandas as pd
# import matplotlib.pyplot as plt

# # ======================
# # Global style (paper, color)
# # ======================
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 10,
#     "axes.labelsize": 10,
#     "axes.titlesize": 10,
#     "legend.fontsize": 9,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "lines.linewidth": 1.8,
#     "axes.linewidth": 0.8,
# })

# # ======================
# # Load data
# # ======================
# dataset_name = "butterflies"
# df = pd.read_csv(f"./{dataset_name}_loss.csv")

# # ======================
# # Figure
# # ======================
# fig, ax = plt.subplots(figsize=(3.5, 2.5))  # 单栏论文尺寸

# ax.plot(
#     df["epoch"],
#     df["loss"],
#     # color="#4E79A7",      # 论文常用蓝
#     color="#F28E2B", # 深橙
#     linestyle="-",
#     label="Training Loss"
# )

# # Labels
# ax.set_xlabel("Epoch")
# ax.set_ylabel("Loss")

# # Title（可选）
# ax.set_title("Training Loss Curve")

# # Grid（淡一点）
# ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

# # Tick style
# ax.tick_params(direction="in", length=4, width=0.8)

# # Legend（如果有多条曲线才需要）
# ax.legend(frameon=False)

# # Layout
# fig.tight_layout()

# # Save
# plt.savefig(f"{dataset_name}_loss.pdf")
# plt.savefig(f"{dataset_name}_loss.png", dpi=300)
# plt.close()
