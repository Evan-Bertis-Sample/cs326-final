from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.table import Table
import time
import functools
import inspect

console = Console()

def timed_banner(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to bind arguments to parameter names
        try:
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arg_strs = []
            for k, v in bound.arguments.items():
                val = repr(v)
                if len(val) > 80:  # limit long values
                    val = val[:77] + "..."
                arg_strs.append((k, val))
        except Exception:
            arg_strs = [("args", str(args)), ("kwargs", str(kwargs))]

        # Build a rich table for arguments
        arg_table = Table(title="Arguments", box=box.MINIMAL_DOUBLE_HEAD)
        arg_table.add_column("Name", style="bold cyan")
        arg_table.add_column("Value", style="white")
        for name, val in arg_strs:
            arg_table.add_row(name, val)

        console.print(Panel.fit(
            f"[bold cyan]{func.__name__}() started[/]",
            box=box.DOUBLE
        ))
        console.print(arg_table)

        # Run function
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dur = time.perf_counter() - t0

        console.print(Panel(
            f"[bold cyan]{func.__name__}() ended[/]\n"
            f"Duration: [green]{dur:.3f}s[/]",
            box=box.DOUBLE
        ))
        return result

    return wrapper
