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

