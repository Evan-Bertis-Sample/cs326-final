# analysis/io.py
from __future__ import annotations
import functools, inspect, time

# basic ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def _short(v: object, maxlen: int = 80) -> str:
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen - 3] + "..."

def banner(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fname = func.__name__
        print(f"{CYAN}[{fname}]{RESET} Begin")

        try:
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for k, v in bound.arguments.items():
                print(f"{CYAN}[{fname}]{RESET} {k} = {_short(v)}")
        except Exception:
            print(f"{CYAN}[{fname}]{RESET} args = {_short(args)}")
            print(f"{CYAN}[{fname}]{RESET} kwargs = {_short(kwargs)}")

        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"{CYAN}[{fname}]{RESET} Complete — took {GREEN}{dt:.3f}s{RESET}\n")
        return result
    return wrapper
