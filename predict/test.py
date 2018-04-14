import numpy as np

a = np.ones((2, 2))
b = np.zeros((2, 2))

print(np.concatenate([a, b], axis=1))
