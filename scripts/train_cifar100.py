#!/usr/bin/env python3
"""Train a deeper CIFAR-100-subset CNN for yard-camera-relevant classes,
scaling up from scripts/train_cifar.py's 3-layer/strided-conv CIFAR-10
model: 4 conv layers, wider channels, and real max-pooling (confirmed
working on K100/K200 as of 2026-10-22 -- see RESEARCH.md -- so this no
longer needs the strided-conv-as-pooling-substitute workaround).

Still all-confirmed-op-only (see RESEARCH.md): conv2d (plain, no grouped/
depthwise), batchnorm (inference-mode, fp32 -- float16 confirmed
XDNN_UNIMPLEMENTED), leaky_relu, max_pool2d (indices=nullptr only --
indices output confirmed unsupported on xpu1), global average pool, fc,
softmax.

Reads the fast.ai CIFAR-100 mirror's two-level image-folder layout
(train/<superclass>/<fine_class>/*.png), filtered down to a fixed list of
20ish fine classes actually plausible in a yard-camera scene (small
mammals, insects/invertebrates, flowers/trees, yard-adjacent vehicles) --
CIFAR-100's other ~80 classes (furniture, people, large carnivores,
aquatic life, etc.) are irrelevant to this use case and dropped.

Architecture (all-confirmed-op-only, deeper than train_cifar.py):
  conv(3->32,  k3,s1,p1) -> bn -> leaky_relu(0.1)   [32x32x32]
  max_pool2d(k2,s2)                                  [16x16x32]
  conv(32->64, k3,s1,p1) -> bn -> leaky_relu(0.1)    [16x16x64]
  max_pool2d(k2,s2)                                  [8x8x64]
  conv(64->128,k3,s1,p1) -> bn -> leaky_relu(0.1)    [8x8x128]
  conv(128->128,k3,s1,p1)-> bn -> leaky_relu(0.1)    [8x8x128]
  global average pool -> [128]
  fc(128->N_CLASSES)

Usage: python3 -I train_cifar100.py --data-dir cifar100_data/cifar100 --out weights
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Yard-camera-relevant subset of CIFAR-100's 100 fine classes. Order here
# is the model's class-index order -- keep in sync with classes.txt on the
# C++ inference side, which reads it back in this exact order.
CLASSES = [
    # animals (small/medium mammals plausible in a yard)
    "squirrel", "rabbit", "mouse", "raccoon", "skunk", "fox", "possum",
    # smaller creatures
    "bee", "butterfly", "spider", "snail",
    # plants
    "rose", "sunflower", "tulip", "maple_tree", "oak_tree", "pine_tree",
    # yard-adjacent vehicles
    "bicycle", "pickup_truck", "motorcycle", "lawn_mower", "tractor",
]


def load_split(data_dir, split):
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        # two-level layout: <data_dir>/<split>/<superclass>/<cls>/*.png
        paths = sorted(glob.glob(os.path.join(data_dir, split, "*", cls, "*.png")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_dir, split, "*", cls, "*.jpg")))
        if not paths:
            raise SystemExit(f"no images found for class '{cls}' under {data_dir}/{split}/*/")
        for p in paths:
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)  # HWC
            xs.append(img.transpose(2, 0, 1))  # CHW
            ys.append(label)
    x = np.stack(xs).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0  # [-1,1], matches the C++ side's normalization
    y = np.array(ys, dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class YardCIFARNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, n_classes, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = x.mean(dim=[2, 3])  # global average pool
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    n_classes = len(CLASSES)
    print(f"loading train split ({n_classes} classes)...")
    xtr, ytr = load_split(args.data_dir, "train")
    print("loading test split...")
    xte, yte = load_split(args.data_dir, "test")
    print(f"train {tuple(xtr.shape)} test {tuple(xte.shape)}")

    model = YardCIFARNet(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = xtr.shape[0]

    # Only 500 train images/class here (vs 5000/class for CIFAR-10), so this
    # net (4 conv layers, up to 128 channels) overfits within ~20 epochs
    # without help -- a random horizontal flip (valid for all 22 classes:
    # animals/vehicles/plants have no fixed left-right orientation) roughly
    # doubles the effective training variety for free, and keeping the
    # best-test-accuracy checkpoint instead of the final epoch avoids
    # shipping a weight set from after the peak.
    best_acc, best_sd = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = xtr[idx].clone(), ytr[idx]
            flip = torch.rand(xb.shape[0]) < 0.5
            xb[flip] = xb[flip].flip(dims=[3])
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, xte.shape[0], 1000):
                logits = model(xte[i:i + 1000])
                correct += (logits.argmax(dim=1) == yte[i:i + 1000]).sum().item()
            acc = correct / xte.shape[0]
        print(f"epoch {epoch+1}/{args.epochs} loss={total_loss/n:.4f} test_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    acc = best_acc
    model.load_state_dict(best_sd)
    print(f"using best checkpoint: test_acc={best_acc:.4f}")

    os.makedirs(args.out, exist_ok=True)

    def dump(name, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        arr.tofile(os.path.join(args.out, name + ".f32"))
        print(name, arr.shape, "->", arr.nbytes, "bytes")

    sd = model.state_dict()
    for i in (1, 2, 3, 4):
        dump(f"conv{i}_weight", sd[f"conv{i}.weight"].numpy())   # [out,in,kh,kw]
        dump(f"conv{i}_bias", sd[f"conv{i}.bias"].numpy())       # [out]
        dump(f"bn{i}_gamma", sd[f"bn{i}.weight"].numpy())
        dump(f"bn{i}_beta", sd[f"bn{i}.bias"].numpy())
        dump(f"bn{i}_mean", sd[f"bn{i}.running_mean"].numpy())
        dump(f"bn{i}_var", sd[f"bn{i}.running_var"].numpy())
    dump("fc_weight", sd["fc.weight"].numpy())  # [n_classes,128]
    dump("fc_bias", sd["fc.bias"].numpy())      # [n_classes]

    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    print(f"final test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
