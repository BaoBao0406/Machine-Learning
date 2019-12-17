import numpy as np

rgen = np.random.RandomState(1)
print(rgen)
w = rgen.normal(loc=1.0, scale=0.01, size=2)

print(w)
g = w[1:]
h = w[0]
print(g)
print(h)