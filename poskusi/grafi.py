import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("cross_validation_results_lstm.csv")

df = df.sort_values("lr")

# --- 2. Line Plot: Learning Rate vs MAE ---
plt.figure(figsize=(5.5, 4))  # narrower width
sns.stripplot(x="lr", y="mae", data=df, jitter=True)
plt.title("Learning Rate vs MSE")
plt.xlabel("Learning Rate")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

df = df.sort_values("weight_decay")

# --- 3. Line Plot: Weight Decay vs MAE ---
plt.figure(figsize=(5.5, 4))  # narrower width
sns.stripplot(x="weight_decay", y="mae", data=df, jitter=True)
plt.title("Weight Decay vs MSE")
plt.xlabel("Weight Decay")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()
