
# --- NumPy compat shim (keeps old Cython code happy) ---

import numpy as _np

for _n, _a in {"int": int, "float": float, "bool": bool}.items():

    if not hasattr(_np, _n):

        setattr(_np, _n, _a)

# -------------------------------------------------------



from BeatNet.BeatNet import BeatNet

import numpy as np, os, sys



if len(sys.argv) < 2:

    print("Usage: python beatnet_demo.py my_audio.wav"); raise SystemExit



wav = os.path.expanduser(sys.argv[1].strip())



estimator = BeatNet(model=1, mode='online', inference_model='PF', plot=[], thread=False)



out = estimator.process(wav)

print("First 10 events:\n", out[:10])



base = os.path.splitext(os.path.basename(wav))[0]

np.savetxt(f"{base}_beats.csv", out, fmt="%.6f,%d",

           header="time_sec,is_downbeat", comments="")



if len(out) > 2:

    ioi = np.diff(out[:,0])

    bpm = 60.0 / np.median(ioi)

    print(f"Estimated tempo ≈ {bpm:.1f} BPM")