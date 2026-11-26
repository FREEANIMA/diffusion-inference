import queue
from .runtime import config_settings, prepared_dir

config = config_settings()
prepared_dir(config)
console = queue.Queue(maxsize=64)