import numpy as np
import matplotlib.pyplot as plt

# Actual function
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)

# Add noise to learning data
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# Draw Diagram to confirm
x = np.linspace(-2, 2, 100)
plt.plot(train_x, train_y, 'o')
plt.plot(x, g(x), linestyle='dashed')
plt.ylim(-1, 2)
plt.show()

# Standardize
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# Setup Learning data matrix
def to_matrix(x):
    return np.vstack([np.ones(x.size), 
                      x,
                      x ** 2,
                      x ** 3, 
                      x ** 4,
                      x ** 5,
                      x ** 6,
                      x ** 7,
                      x ** 8, 
                      x ** 9,
                      x ** 10,
                      ]).T

X = to_matrix(train_z)

# Variable initialize
theta = np.random.randn(X.shape[1])

# Predict function
def f(x):
    return np.dot(x, theta)

def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# Learning rate
ETA = 1e-4

# diff
diff = 1

# repeat learning
error = E(X, train_y)
while diff > 1e-6:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# Diagram result
z = standardize(x)
plt.plot(train_z, train_y, 'o')
plt.plot(z, f(to_matrix(z)))
plt.show()

# Save non-standardize variable and initialize again
theta1 = theta
theta = np.random.randn(X.shape[1])

# Initialize varibale
LAMBDA = 1

# diff
diff = 1

# repeat learning (with standardize)
error = E(X, train_y)
while diff > 1e-6:
    # Standardize the data
    reg_term = LAMBDA * np.hstack([0, theta[1:]])
    # Update variable
    theta = theta - ETA * (np.dot(f(X) - train_y, X) + reg_term)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# Draw Diagram
plt.plot(train_z, train_y, 'o')
plt.plot(z, f(to_matrix(z)))
plt.show()

# Savev standardize variable
theta2 = theta

plt.plot(train_z, train_y, 'o')

# No standardize diagram result
theta = theta1
plt.plot(z, f(to_matrix(z)), linestyle='dashed')

# Standardize diagram result
theta = theta2
plt.plot(z, f(to_matrix(z)))

plt.show()