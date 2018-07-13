import time

class _State:
    TRACER_ENABLED = True


_state = _State()


def enable_tracing(enabled: bool):
    _state.TRACER_ENABLED = enabled


def traced(method):
    def decorator(*args, **kw):
        if not _state.TRACER_ENABLED:
            return method(*args, **kw)
        name = method.__name__
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print('{}  - finished in {:.2f} ms'.format(name, ((end - start) * 1000)))
        return result
    return decorator
