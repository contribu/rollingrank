
import numpy as np
import rollingrank_native

def rollingrank(x, *args, **kwargs):
    if hasattr(x, 'to_numpy'):
        x = x.to_numpy()
    if isinstance(x, list):
        x = np.array(x)
    return rollingrank_native.rollingrank(x, *args, **kwargs)
