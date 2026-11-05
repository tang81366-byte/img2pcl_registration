from functools import wraps
from time import perf_counter

# 记录函数执行耗时
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            print(f"{func.__name__} 耗时：{elapsed:.3f} 秒")
    return wrapper