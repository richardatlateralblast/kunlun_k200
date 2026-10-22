#!/usr/bin/env python3
"""Evaluate a built cifar10_live/cifar100_live binary against real held-out
test images with known ground truth (not just "runs without crashing") --
same style of check used for cifar10_live's 59.0%-on-device-accuracy result
(see RESEARCH.md "Real recognition: CIFAR-10 classifier"), now scripted so
it's reusable instead of ad hoc.

Picks --n-per-class test images per class (from the fast.ai mirror's
image-folder layout -- flat for CIFAR-10, <split>/<superclass>/<class>/ for
CIFAR-100, both handled via a glob), converts each to the raw RGB24 32x32
stream the binary reads from stdin, runs the binary once over the whole
batch, and compares each predicted top class against the known label.

Usage:
  python3 scripts/eval_cifar_live.py \\
    --binary demos/cifar100_live --weights-dir demos/cifar100_weights \\
    --data-dir cifar100_data/cifar100 --n-per-class 20
"""
import argparse, glob, os, subprocess, sys
import numpy as np
from PIL import Image


def load_classes(weights_dir):
    with open(os.path.join(weights_dir, "classes.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def pick_test_images(data_dir, cls, n):
    # CIFAR-100 fast.ai layout: <data_dir>/test/<superclass>/<cls>/*.png
    paths = sorted(glob.glob(os.path.join(data_dir, "test", "*", cls, "*.png")))
    if not paths:
        # CIFAR-10 fast.ai layout: <data_dir>/test/<cls>/*.png
        paths = sorted(glob.glob(os.path.join(data_dir, "test", cls, "*.png")))
    if not paths:
        sys.exit(f"no test images found for class '{cls}' under {data_dir}/test/")
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=20)
    args = ap.parse_args()

    classes = load_classes(args.weights_dir)
    print(f"{len(classes)} classes: {classes}")

    frames = bytearray()
    truth = []
    for cls in classes:
        for p in pick_test_images(args.data_dir, cls, args.n_per_class):
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)  # HWC, 32x32x3
            if img.shape != (32, 32, 3):
                sys.exit(f"unexpected shape {img.shape} for {p}, expected (32,32,3)")
            frames += img.tobytes()
            truth.append(cls)

    n = len(truth)
    print(f"{n} held-out test images ({args.n_per_class}/class), running {args.binary}...")

    proc = subprocess.run(
        [args.binary, "--frames", str(n), "--weights-dir", args.weights_dir],
        input=bytes(frames), capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout.decode(errors="replace"))
        print(proc.stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(f"{args.binary} exited {proc.returncode}")

    preds = []
    for line in proc.stdout.decode(errors="replace").splitlines():
        if line.startswith("frame "):
            # "frame 0: squirrel (41.2%) [...]"
            pred = line.split(":", 1)[1].strip().split(" ", 1)[0]
            preds.append(pred)

    if len(preds) != n:
        sys.exit(f"expected {n} prediction lines, got {len(preds)} -- binary output:\n" +
                  proc.stdout.decode(errors="replace"))

    correct = sum(1 for p, t in zip(preds, truth) if p == t)
    confusions = {}
    for p, t in zip(preds, truth):
        if p != t:
            confusions[(t, p)] = confusions.get((t, p), 0) + 1

    print(f"\naccuracy: {correct}/{n} = {correct/n:.4f}")
    if confusions:
        print("top confusions (true -> predicted: count):")
        for (t, p), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {t} -> {p}: {c}")


if __name__ == "__main__":
    main()
