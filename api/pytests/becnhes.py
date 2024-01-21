from reference.packed import permanova as packed_permanova
from reference.square import permanova as square_permanova
import timeit
import pandas as pd

def time(fs: list[str]):
    res = pd.Series({f:timeit.timeit(f, globals=globals()) for f in fs})
    for scale, unit in [
        (1e-9, 'ns'),
        (1e-6, 'us'),
        (1e-3, 'ms'),
        (   1,  's')
    ]:
        if res.min() < scale:
            res = (res / scale).round(2).astype(str) + ' ' + unit
            break
    res = res.to_frame().reset_index()
    res.columns = ['call', 'time']
    print(res.to_string(index=False))

def a(x):
    return x**42
def b(x):
    return x*42

time([
    'a(3)',
    'b(42)'
])
