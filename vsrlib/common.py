import os
import sys

# Reduce CUDA allocator fragmentation for long tiled runs. Must be set before the
# first CUDA allocation — every torch-importing module in this package imports
# common first, so this runs before torch loads — and never overrides a
# user-provided allocator config.
if sys.platform.startswith("linux") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

# Repo root (this file lives one level below it), same value run.py computed
# when it owned these globals.
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp = os.path.join(root, "_temp")
