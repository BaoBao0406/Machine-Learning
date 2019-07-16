
from matplotlib import pyplot as plt
import numpy as np

X = np.linspace(-np.pi, np.pi, 200, endpoint=True)
C, S = np.cos(X), np.sin(X)

plt.figure(num='sin', figsize=(16, 4))
plt.plot(X, S)
plt.figure(num='cos', figsize=(16, 4))
plt.plot(X, C)
plt.figure(num='sin')
plt.plot(X, C)
print(plt.figure(num='sin').number)
print(plt.figure(num='cos').number)