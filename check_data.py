import torch
import glob

for f in sorted(glob.glob("Data/*.pt")):
    data = torch.load(f, weights_only=False)
    if isinstance(data, dict):
        data = data.get('frame_data', data)
    if torch.isnan(data).any():
        print(f"NaN found: {f}")
    if data.min() < -10 or data.max() > 10:
        print(f"Bad range: {f} | min={data.min():.2f}, max={data.max():.2f}")

print("Data check complete!")
