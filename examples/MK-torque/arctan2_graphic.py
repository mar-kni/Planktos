import numpy as np
import matplotlib.pyplot as plt

# Range of y/x values
ratios = np.linspace(-10, 10, 500)

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)

# Utility function to add reference lines
def add_refs(ax):
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.axhline(np.pi/2, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-np.pi/2, color="gray", linestyle="--", linewidth=1)
    ax.axhline(np.pi, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-np.pi, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-3*np.pi/2, 3*np.pi/2)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("y/x")
    ax.set_ylabel(r"$\theta$")

# 1) x < 0, y > 0
axs[0,0].plot(ratios, np.arctan2(np.abs(ratios), -1), color="red")
axs[0,0].set_title("x < 0, y > 0")
add_refs(axs[0,0])

# 2) x > 0, y > 0
axs[0,1].plot(ratios, np.arctan2(np.abs(ratios), 1), color="blue")
axs[0,1].set_title("x > 0, y > 0")
add_refs(axs[0,1])

# 3) x < 0, y < 0
axs[1,0].plot(ratios, np.arctan2(-np.abs(ratios), -1), color="purple")
axs[1,0].set_title("x < 0, y < 0")
add_refs(axs[1,0])

# 4) x > 0, y < 0
axs[1,1].plot(ratios, np.arctan2(-np.abs(ratios), 1), color="green")
axs[1,1].set_title("x > 0, y < 0")
add_refs(axs[1,1])

plt.suptitle(r"$\mathrm{atan2}(y,x)$ vs $y/x$ in four quadrants", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("examples/MK-torque/arctan2_graphic.png", dpi=300)
plt.show()

