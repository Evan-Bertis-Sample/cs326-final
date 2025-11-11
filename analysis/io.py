from rich.console import Console
from rich import box
from rich.panel import Panel
import time
import functools

console = Console()

def timed_banner(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        console.print(Panel(f"[bold cyan]{func.__name__}() started[/]", box=box.DOUBLE))
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        dur = t1 - t0
        console.print(Panel(f"[bold cyan]{func.__name__}() ended[/]\nDuration: [green]{dur:.3f}s[/]", box=box.DOUBLE))
        return result
    return wrapper
