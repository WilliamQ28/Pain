# verify_cuda.py
import sys

try:
    import torch
except Exception as e:
    print("[ERR] Could not import torch:", e)
    sys.exit(1)

print("torch:", getattr(torch, "__version__", "unknown"))
avail = torch.cuda.is_available()
print("cuda_available:", avail)

if avail:
    try:
        print("cuda_runtime:", torch.version.cuda)
        print("device_count:", torch.cuda.device_count())
        print("device_0:", torch.cuda.get_device_name(0))
        print("capability_0:", ".".join(map(str, torch.cuda.get_device_capability(0))))
    except Exception as e:
        print("[WARN] CUDA visible but query failed:", e)
else:
    print("[HINT] If this should be True, ensure you installed the CUDA build of torch and that this venv’s python.exe is set to use the NVIDIA GPU.")
