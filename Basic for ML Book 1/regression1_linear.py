import numpy as np
import matplotlib.pyplot as plt

# read click.csv data
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
#print(train)
train_x = train[:,0]
#print(train_x)
train_y = train[:,1]

# diagram
plt.plot(train_x, train_y, 'o')
plt.show()

# random variables
theta0 = np.random.rand()
theta1 = np.random.rand()

# predict function
def f(x):
    return theta0 + theta1 * x

# target function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# Standardize
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)
plt.plot(train_z, train_y, 'o')
plt.show()

# Learning rate
ETA = 1e-3

# Difference
diff = 1

# Update times
count = 0

# Repeat learning
error = E(train_z, train_y)
while diff > 1e-2:
    # Update result for variable
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # Update variable
    theta0 = tmp0
    theta1 = tmp1
    # Calculate the difference with preivous
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # Export diary
    count += 1
    log = '{} times: theta0 = {:.3f}, theta1 = {:.3f}, diff = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()
    