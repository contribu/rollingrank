
import numba
import numpy as np
import rollingrank_native

def rollingrank(x, *args, **kwargs):
    if hasattr(x, 'to_numpy'):
        x = x.to_numpy()
    if isinstance(x, list):
        x = np.array(x)
    return rollingrank_native.rollingrank(x, *args, **kwargs)

def rci(x, *args, **kwargs):
    if hasattr(x, 'to_numpy'):
        x = x.to_numpy()
    if isinstance(x, list):
        x = np.array(x)
    if x.dtype == np.dtype('float16'):
        x = x.astype('float32')
    return rollingrank_native.rci(x, *args, **kwargs)

@numba.njit
def rci_reference(x, window=5):
    result = x.copy()
    result[:window - 1] = np.nan
    t = np.arange(window)
    for end in range(window - 1, result.size):
        start = end - window + 1
        target = x[start:end + 1]
        target_argsort = np.argsort(target)
        result[end] = np.corrcoef(target_argsort, t)[0, 1]
    return result
