import numpy as np
import src

x = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3])
y = src.rollingrank(x, window=3)
print(y)

y = src.rollingrank(x, window=3, pct=True)
print(y)
