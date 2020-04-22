## rollingrank

rollingrank is a fast implementation of rolling rank transformation (described as the following code).

```python
import pandas as pd

# x is numpy array
def rollingrank(x, window=None):
    def to_rank(x):
        # result[i] is the rank of x[i] in x
        return np.sum(np.less(x, x[-1]))
    return pd.Series(x).rolling(window).apply(to_rank).values
```

## Motivation

Rolling rank is a good tool to create features for time series prediction.
However, rolling rank was not easy to use in python.
There were no exact methods to do it.
The simple implementation using pandas and numpy is too slow.

## Performance

|Implementation|Complexity|
|:-:|:-:|
|rollingrank|O(n * log(w))|
|pandas rolling + numpy|O(n * w)|

n: input length
w: rolling window size

## Install

```bash
pip install rollingrank
```

## Example

```python
import numpy as np
import rollingrank

x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
y = rollingrank.rollingrank(x, window=3)
print(y)
# [nan nan  2.  1.  0.  1.  2.]

y = rollingrank.rollingrank(x, window=3, pct=True)
print(y)
# [nan nan 1.  0.5 0.  0.5 1. ]
```

## Kaggle Example

https://www.kaggle.com/bakuage/rollingrank-example

## Development

test

```bash
python -m unittest discover tests
```

build/upload

```bash
python setup.py sdist
twine upload --repository pypitest dist/*
twine upload --repository pypi dist/*
```

## TODO

- support axis
