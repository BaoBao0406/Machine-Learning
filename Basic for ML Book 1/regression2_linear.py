import numpy as np
import matplotlib.pyplot as plt

# read click.csv data
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
#print(train)
train_x = train[:,0]
#print(train_x)
train_y = train[:,1]

# Standardize
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

theta = np.random.rand(3)

# Setup data's matrix
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

# predict function
def f(x):
    return np.dot(x, theta)

# target function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# Learning rate
ETA = 1e-3

# Difference
diff = 1

# repeat learning
error = E(X, train_y)
while diff > 1e-2:
    # update variable
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # calculate the difference between previous
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()

# MSE
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

# Convert to randomize
theta = np.random.rand(3)

# MSE error
errors = []

# Error diff
diff = 1

# Repeat learning
errors.append(MSE(X, train_y))
while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

# diagram
x = np.arange(len(errors))

plt.plot(x, errors)
plt.show()

# Use variables to randomize
theta = np.random.rand(3)

# MSE history record
errors = []

# MSE difference
diff = 1

# Repeat learning
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # Prepare data for randomize
    p = np.random.permutation(X.shape[0])
    # Use data as randomize method for regression
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    # Calculate the difference
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()