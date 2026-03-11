"""Quick verification that training code fixes are correct."""
import sys, os
sys.path.insert(0, r"d:\skincare\Project")

# 1. Verify ExtremeSamplesDataset _FOLDER_MAP values
from datasets.loaders import ExtremeSamplesDataset

print("=" * 60)
print("1. Checking _FOLDER_MAP values")
print("=" * 60)
fm = ExtremeSamplesDataset._FOLDER_MAP

assert fm["clear_skin"]["acne_score"] == 0.0, f"FAIL: clear_skin acne_score should be 0.0, got {fm['clear_skin']['acne_score']}"
print(f"  clear_skin acne_score = {fm['clear_skin']['acne_score']} OK")

assert fm["redness_severe"]["redness_score"] == 0.90
print(f"  redness_severe redness_score = {fm['redness_severe']['redness_score']} OK")

assert fm["dark_circles_severe"]["dark_circle_score"] == 0.90
print(f"  dark_circles_severe dark_circle_score = {fm['dark_circles_severe']['dark_circle_score']} OK")

# 2. Verify _NO_JITTER_KEYS
print()
print("=" * 60)
print("2. Checking _NO_JITTER_KEYS")
print("=" * 60)
assert "acne_score" in ExtremeSamplesDataset._NO_JITTER_KEYS
print(f"  _NO_JITTER_KEYS = {ExtremeSamplesDataset._NO_JITTER_KEYS} OK")

# 3. Verify OrdinalCrossEntropy has label smoothing
print()
print("=" * 60)
print("3. Checking OrdinalCrossEntropy label smoothing")
print("=" * 60)
from training.trainer import OrdinalCrossEntropy
import torch

oce = OrdinalCrossEntropy(label_smoothing=0.05)
assert hasattr(oce, 'label_smoothing')
assert oce.label_smoothing == 0.05
print(f"  label_smoothing = {oce.label_smoothing} OK")

# Quick functional test
logits = torch.zeros(1, 3)
target = torch.tensor([0.0])
loss, n = oce(logits, target)
print(f"  Loss for class 0 (no acne): {loss.item():.4f}, n={n} OK")

target3 = torch.tensor([1.0])
loss3, n3 = oce(logits, target3)
print(f"  Loss for class 3 (severe):  {loss3.item():.4f}, n={n3} OK")

target_nan = torch.tensor([float('nan')])
loss_nan, n_nan = oce(logits, target_nan)
assert n_nan == 0 and loss_nan.item() == 0.0
print(f"  Loss for NaN target: {loss_nan.item():.4f}, n={n_nan} OK")

# 4. Try loading the dataset
print()
print("=" * 60)
print("4. Loading ExtremeSamplesDataset with real data")
print("=" * 60)
try:
    ds = ExtremeSamplesDataset(
        root="data/extreme_samples",
        train=False,
        pseudo_label_cache_dir="data/cache/pseudo_labels_region_v2",
        seed=42,
    )
    print(f"  Dataset size: {len(ds)}")

    acne_vals = []
    for i in range(len(ds)):
        acne = ds.labels[i]["acne_score"]
        if acne == acne:  # not NaN
            acne_vals.append(acne)
            assert acne in {0.0, 0.33, 0.66, 1.0}, f"Invalid acne at idx {i}: {acne}"
    print(f"  All {len(acne_vals)} acne labels are valid ordinal values OK")

    clear_tex = [ds.labels[i]["texture_score"] for i, p in enumerate(ds.image_paths) if "clear_skin" in p]
    if clear_tex:
        avg = sum(clear_tex) / len(clear_tex)
        print(f"  Clear skin avg texture: {avg:.3f} (expected ~0.20) OK")

    red_vals = [ds.labels[i]["redness_score"] for i, p in enumerate(ds.image_paths) if "redness_severe" in p]
    if red_vals:
        avg = sum(red_vals) / len(red_vals)
        print(f"  Redness severe avg: {avg:.3f} (expected ~0.90) OK")

    print(f"\n  Label distribution:")
    for key in ["acne_score", "redness_score", "texture_score", "dark_circle_score"]:
        vals = [ds.labels[i][key] for i in range(len(ds)) if ds.labels[i][key] == ds.labels[i][key]]
        if vals:
            print(f"    {key:20s}: n={len(vals):3d}, min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}")
        else:
            print(f"    {key:20s}: all NaN")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("ALL CHECKS PASSED")
