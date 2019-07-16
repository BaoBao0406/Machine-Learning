from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 200)
C, S = np.cos(x), np.sin(x)
plt.plot(x, C, color='blue', linewidth=2.0, linestyle='-')
plt.plot(x, S, color='red', linewidth=2.0, linestyle='-')
plt.xlim(x.min() * 1.1, x.max() * 1.1)
plt.ylim(C.min() * 1.1, C.max() * 1.1)
plt.show()

x = np.linspace(-np.pi, np.pi, 200)
C, S = np.cos(x), np.sin(x)
plt.plot(x, C, color='blue', linewidth=2.0, linestyle='-')
plt.plot(x, S, color='red', linewidth=2.0, linestyle='-')
plt.xlim(x.min() * 1.1, x.max() * 1.1)
plt.ylim(C.min() * 1.1, C.max() * 1.1)
plt.xticks((-np.pi, -np.pi/2, np.pi/2, np.pi), \
           (r'$-\pi$', r'$-\pi/2$', r'$+\pi/2$', r'$+\pi$'))
plt.yticks([-1, -0.5, 0, 0.5, 1])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.legend(loc='upper left')
t = 2 * np.pi / 3
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=1.5, linestyle='--')
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
plt.show()

