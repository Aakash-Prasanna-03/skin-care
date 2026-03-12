"""
merge_checkpoints.py — Merge acne head weights from one checkpoint with other
head weights from another. Solves the multi-task forgetting problem where acne
detection degrades while other features improve.

Usage:
    python merge_checkpoints.py \
        --acne_checkpoint checkpoints/checkpoint_epoch010.pth \
        --base_checkpoint checkpoints/best_model.pth \
        --output checkpoints/merged_model.pth
"""

import argparse
import torch


# Keys belonging to the acne-specific head and the full-face encoder
# (the region encoder that feeds into the acne head)
ACNE_KEY_PREFIXES = (
    "acne_head.",
    "full_face_encoder.",
)


def merge(acne_path: str, base_path: str, output_path: str) -> None:
    acne_ckpt = torch.load(acne_path, map_location="cpu")
    base_ckpt = torch.load(base_path, map_location="cpu")

    acne_sd = acne_ckpt["state_dict"]
    base_sd = base_ckpt["state_dict"]

    merged_sd = {}
    acne_keys = []
    base_keys = []

    for key in base_sd:
        if any(key.startswith(prefix) for prefix in ACNE_KEY_PREFIXES):
            merged_sd[key] = acne_sd[key]
            acne_keys.append(key)
        else:
            merged_sd[key] = base_sd[key]
            base_keys.append(key)

    # Build merged checkpoint
    merged_ckpt = {
        "epoch": base_ckpt.get("epoch", -1),
        "val_loss": base_ckpt.get("val_loss", float("inf")),
        "state_dict": merged_sd,
        "model_config": base_ckpt.get("model_config", {}),
        "train_config": base_ckpt.get("train_config", {}),
        "merge_info": {
            "acne_source": acne_path,
            "base_source": base_path,
            "acne_key_count": len(acne_keys),
            "base_key_count": len(base_keys),
        },
    }

    torch.save(merged_ckpt, output_path)
    print(f"Merged checkpoint saved to: {output_path}")
    print(f"  Acne head + full_face_encoder from: {acne_path} ({len(acne_keys)} keys)")
    print(f"  Everything else from: {base_path} ({len(base_keys)} keys)")


def main():
    parser = argparse.ArgumentParser(description="Merge acne head from one checkpoint with other heads from another")
    parser.add_argument("--acne_checkpoint", default="checkpoints/checkpoint_epoch010.pth",
                        help="Checkpoint with good acne detection")
    parser.add_argument("--base_checkpoint", default="checkpoints/best_model.pth",
                        help="Checkpoint with good redness/texture/dark_circle detection")
    parser.add_argument("--output", default="checkpoints/merged_model.pth",
                        help="Output path for merged checkpoint")
    args = parser.parse_args()
    merge(args.acne_checkpoint, args.base_checkpoint, args.output)


if __name__ == "__main__":
    main()
