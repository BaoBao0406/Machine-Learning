from matplotlib import pyplot as plt
import numpy as np
"""
# Example 1
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
plt.subplot(1, 2, 1)
n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)
X, Y = np.meshgrid(x, y)
plt.imshow(f(X, Y), cmap='hot', origin='low')
plt.colorbar(shrink=.83)
plt.xticks(())
plt.yticks(())

# Example 2
plt.subplot(1, 2, 2)
n = 20
Z = np.ones(n)
Z[-1] *= 2
plt.pie(Z, explode= Z * .05, colors = ['%f' % (i / float(n)) for i in range(n)])
plt.axis('equal')
plt.xticks(())
plt.yticks(())

# Example 3
ax = plt.subplot(1, 2, 1)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
ax.set_xticklabels(())
ax.set_yticklabels(())
"""
# Example 4
ax = plt.subplot(1, 2, 2, polar=True)
N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r/10.))
    bar.set_alpha(0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
