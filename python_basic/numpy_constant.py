import numpy as np
from numpy import nan as NAN

b = np.array([0, 1, 2, 3, 4, 5])
a = np.array([0, 1, 2, 3, 4, 5])
c = b/a
print(c)
print(np.isnan(c))
print(np.isnan(NAN))
print(NAN)

value = np.inf
print(value)
