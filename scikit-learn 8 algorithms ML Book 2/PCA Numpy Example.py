import numpy as np

# PCA using numpy
A = np.array([[3, 2000], [2, 3000], [4, 5000], [5, 8000], [1, 2000]], dtype='float')
# Standardize data
mean = np.mean(A, axis=0)
norm = A - mean
# Scaler data
scope = np.max(norm, axis=0) - np.min(norm, axis=0)
norm = norm / scope
print(norm)
# Using singular value decomposition to find Eigenvector
U, S, V = np.linalg.svd(np.dot(norm.T, norm))
print()
print(U)
print()
U_reduce = U[:, 0].reshape(2, 1)
print(U_reduce)
print()
R = np.dot(norm, U_reduce)
print(R)

# Reduction using PCA
Z = np.do(R, U_reduce.T)
B = np.multiply(Z, scope) + mean
