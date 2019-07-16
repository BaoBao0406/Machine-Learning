import numpy as np
from matplotlib import pyplot as plt
n_dots = 40
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).ravel()

# Add some noise
y += 0.2 * np.random.rand(n_dots) - 0.1

# Train module
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)

# Create enough point and predict
T = np.array(np.linspace(0, 5, 100)).reshape(-1, 1)
y_pred = knn.predict(T)
print(knn.score(X, y))

plt.figure(figsize=(16, 10))
plt.scatter(X, y, c='g', label='data', s=100)
plt.plot(T, y_pred, c='k', label='prediction', lw=4)
plt.axis('tight')
plt.title('KNeighborsRegressor (k = %i)' % k)
plt.show()
"""
print(np.linspace(0, 5, 100)[:, np.newaxis])
print(np.array(np.linspace(0, 5, 100)).reshape(-1, 1))"""
print(T)
print(y_pred)