import numpy as np
import matplotlib.pyplot as plt
 
# Load learning data
train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:,2]

# Draw Diagram
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.axis('scaled')
plt.show()

# weight initialize
w = np.random.rand(2)

# Identify function
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# Count for repeat
epoch = 10

# Update times
count = 0

# Learning weight
for _ in range(epoch):
    print(_)
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x
            # Output log
            count += 1
            print('{} times: w = {}'.format(count, w))

x1 = np.arange(0, 500)

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
print(w)
plt.show()

