import numpy as np
import rollingrank

x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
y = rollingrank.rollingrank(x, window=3)
print(y)

y = rollingrank.rollingrank(x, window=3, pct=True)
print(y)
