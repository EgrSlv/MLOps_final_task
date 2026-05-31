import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

existing = np.random.normal(loc=3.5, scale=0.4, size=500000)
improved = np.random.normal(loc=2.0, scale=0.4, size=500000)

print(f"Existing system: mean={existing.mean():.4f}, std={existing.std():.4f}")
print(f"Improved system: mean={improved.mean():.4f}, std={improved.std():.4f}")
print(f"Improvement: {(existing.mean() - improved.mean()) / existing.mean() * 100:.1f}%")

t_stat, p_value = stats.ttest_ind(improved, existing, alternative="less")
print(f"\nOne-sided t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.15f}")
print(f"  Alpha: 0.05")
print(f"  Reject H0: {p_value < 0.05}")

plt.figure(figsize=(10, 6))
sns.kdeplot(existing, label="Existing system", fill=True, color="red")
sns.kdeplot(improved, label="Improved system", fill=True, color="green")
plt.title("System Response Time Comparison")
plt.xlabel("Response Time (seconds)")
plt.ylabel("Observations")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

ax = plt.gca()
ymin, ymax = ax.get_ylim()
num_ticks = 5
new_yticks = np.linspace(ymin, ymax, num_ticks)
new_yticklabels = [
    f"{int((tick / ymax) * 100)}%" if ymax != 0 else "0%"
    for tick in new_yticks
]
ax.set_yticks(new_yticks)
ax.set_yticklabels(new_yticklabels)

plt.savefig("../.misc/images/image3.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to .misc/images/image3.png")
