from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
"""
# Example 1
plt.figure(figsize=(18, 4))
plt.subplot(1, 1, 1)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2, 2, 1)', ha='center', va='center', size=20, alpha=.5)

plt.subplot(2, 2, 2)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2, 2, 2)', ha='center', va='center', size=20, alpha=.5)

plt.subplot(2, 2, 3)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2, 2, 3)', ha='center', va='center', size=20, alpha=.5)

plt.subplot(2, 2, 4)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(2, 2, 4)', ha='center', va='center', size=20, alpha=.5)

plt.tight_layout()
plt.show()

# Example 2
plt.figure(figsize=(18,4))
G = gridspec.GridSpec(3,3)

axes_1 = plt.subplot(G[0,:])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 1', ha='center', va='center', size=24, alpha=.5)

axes_2 = plt.subplot(G[1:, 0])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 2', ha='center', va='center', size=24, alpha=.5)

axes_3 = plt.subplot(G[1:, -1])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 3', ha='center', va='center', size=24, alpha=.5)

axes_4 = plt.subplot(G[1, -2])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 4', ha='center', va='center', size=24, alpha=.5)

axes_5 = plt.subplot(G[-1, -2])
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'Axes 5', ha='center', va='center', size=24, alpha=.5)
plt.tight_layout()
plt.show()
"""
# Example 3
plt.figure(figsize=(18,4))
plt.axes([.1, .1, .8, .8])
plt.xticks(())
plt.yticks(())
plt.text(.2, .5, 'axes([0.1, 0.1, .8, .8])', ha='center', va='center', size=20, alpha=.5)
plt.axes([.5, .5, .3, .3])
plt.xticks(())
plt.yticks(())
plt.text(.5, .5, 'axes([.5, .5, .3, .3])', ha='center', va='center', size=16, alpha=.5)
plt.show()