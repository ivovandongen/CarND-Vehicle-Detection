import time


def traced(method):
    def decorator(*args, **kw):
        name = method.__name__
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print('{}  - finished in {:.2f} ms'.format(name, ((end - start) * 1000)))
        return result
    return decorator
