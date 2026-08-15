import argparse, time, yaml, torch, numpy as np
from pathlib import Path
from models.nafnet_lite import build_model

ap=argparse.ArgumentParser()
ap.add_argument("--input_dir",required=True)
ap.add_argument("--checkpoint",default="checkpoints/latest_model.pth")
ap.add_argument("--model_config",default="configs/model.yaml")
args=ap.parse_args()
cfg=yaml.safe_load(open(args.model_config))
device="cuda" if torch.cuda.is_available() else "cpu"
model=build_model(cfg).to(device).eval()
ckpt=torch.load(args.checkpoint,map_location=device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
files=sorted(Path(args.input_dir).glob("*.npy"))
if not files:
    raise SystemExit("No .npy files found.")
a=np.load(files[0]).astype(np.float32)
if a.ndim==2: a=a[None]
x=torch.from_numpy(a).unsqueeze(0).to(device)
with torch.no_grad():
    for _ in range(5): model(x)
if device=="cuda": torch.cuda.synchronize()
t=time.perf_counter()
with torch.no_grad():
    for p in files:
        a=np.load(p).astype(np.float32)
        if a.ndim==2: a=a[None]
        xx=torch.from_numpy(a).unsqueeze(0).to(device)
        model(xx)
if device=="cuda": torch.cuda.synchronize()
elapsed=time.perf_counter()-t
print(f"Images: {len(files)}")
print(f"Total model-loop time: {elapsed:.4f}s")
print(f"Average per image: {elapsed/len(files):.6f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
