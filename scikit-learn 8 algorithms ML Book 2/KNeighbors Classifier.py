from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np

# Create data
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

# Draw Data
#plt.figure(figsize=(16, 10), dpi=144)
c = np.array(centers)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
#plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')

from sklearn.neighbors import KNeighborsClassifier
# Train module
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# Perform predit
X_sample = [0, 2]
#print(np.array(X_sample).reshape(-1, 1))
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)

# Draw Diagram
plt.figure(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')
plt.scatter(X_sample[0][0], X_sample[0][1], marker='x', s=100, cmap='cool')

for i in neighbors[0]:
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)

