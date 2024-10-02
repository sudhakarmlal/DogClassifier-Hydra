import sys
from pathlib import Path
from functools import wraps

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

def setup_logger(log_file):
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file, rotation="10 MB")

def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise
    return wrapper
def log_metrics_table(metrics: dict, title: str):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    rich_print(table)

 

def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )