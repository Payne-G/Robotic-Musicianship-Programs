# --- NumPy compat shim (for older Cython code) ---
import numpy as _np
for _n, _a in {"int": int, "float": float, "bool": bool}.items():
    if not hasattr(_np, _n):
        setattr(_np, _n, _a)
# -------------------------------------------------

from BeatNet.BeatNet import BeatNet
import numpy as np

# Create BeatNet in live/stream mode
estimator = BeatNet(
    model=1,
    mode='stream',           # live stream mode
    inference_model='PF',
    plot=['activations','beats'],                 
    thread=False             # keep it single-threaded for simplicity
)

print("Listening for beats... Press Ctrl+C to stop.")

try:
    for beat_time, is_downbeat in estimator.process(None):
        print(f"There was a beat")
except KeyboardInterrupt:
    print("\nStopped live detection.")