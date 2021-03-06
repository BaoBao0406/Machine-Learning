from matplotlib import pyplot as plt
import numpy as np

# Example 1
def tickline():
    plt.xlim(0, 10), plt.ylim(-1, 1), plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # Set label size
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
    ax.plot(np.arange(11), np.zeros(11))
    return ax

locators = [
            'plt.NullLocator()',
            'plt.MultipleLocator(base=1.0)',
            'plt.FixedLocator(locs=[0, 2, 8, 9, 10])',
            'plt.IndexLocator(base=3, offset=1)',
            'plt.LinearLocator(numticks=5)',
            'plt.LogLocator(base=2, subs=[1.0])',
            'plt.MaxNLocator(nbins=3, steps=[1, 3, 5, 7, 9, 10])',
            'plt.AutoLocator()']

n_locators = len(locators)

# Calculate diagram size
size = 1024, 60 * n_locators
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)

for i, locator in enumerate(locators):
    plt.subplot(n_locators, 1, i + 1)
    ax = tickline()
    ax.xaxis.set_major_locator(eval(locator))
    
    plt.text(5, 0.3, locator[3:], ha='center', size=16)

plt.subplots_adjust(bottom=.1, top=.99, left=.01, right=.99)
plt.show()