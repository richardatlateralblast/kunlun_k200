#!/usr/bin/env python3
"""Train a tiny CIFAR-10 CNN matching exactly the ops confirmed working on
K100/K200 (xpu1): plain conv2d (no grouped/depthwise), batchnorm, leaky_relu,
global average pool, fc, softmax. No skip connections, no pooling layers
(downsampling via strided conv instead, to avoid a pooling op never tested
on this chip this session).

Reads the public CIFAR-10 dataset from an image-folder layout
(train/<class>/*.png, test/<class>/*.png -- the fast.ai S3 mirror's format),
not the original pickled-batch format, since that mirror is dramatically
faster to fetch than the official source from this network.

Architecture (all-confirmed-op-only, see RESEARCH.md):
  conv(3->16, k3, s1) -> bn -> leaky_relu(0.1)   [32x32x16]
  conv(16->32, k3, s2) -> bn -> leaky_relu(0.1)  [16x16x32]
  conv(32->64, k3, s2) -> bn -> leaky_relu(0.1)  [8x8x64]
  global average pool -> [64]
  fc(64->10)

Input normalization is deliberately trivial (scale to [-1,1], no per-channel
mean/std) so the C++ inference side doesn't need to reproduce a fussier
normalization exactly.

Usage: python3 -I train_cifar.py --data-dir cifar_data/cifar10 --out weights
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_split(data_dir, split):
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        paths = sorted(glob.glob(os.path.join(data_dir, split, cls, "*.png")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(data_dir, split, cls, "*.jpg")))
        for p in paths:
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)  # HWC
            xs.append(img.transpose(2, 0, 1))  # CHW
            ys.append(label)
    x = np.stack(xs).astype(np.float32) / 255.0
    x = x * 2.0 - 1.0  # [-1,1], matches the C++ side's normalization
    y = np.array(ys, dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class TinyCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = x.mean(dim=[2, 3])  # global average pool
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("loading train split...")
    xtr, ytr = load_split(args.data_dir, "train")
    print("loading test split...")
    xte, yte = load_split(args.data_dir, "test")
    print(f"train {tuple(xtr.shape)} test {tuple(xte.shape)}")

    model = TinyCIFARNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = xtr.shape[0]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]
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

    os.makedirs(args.out, exist_ok=True)

    def dump(name, arr):
        arr = np.ascontiguousarray(arr.astype(np.float32))
        arr.tofile(os.path.join(args.out, name + ".f32"))
        print(name, arr.shape, "->", arr.nbytes, "bytes")

    sd = model.state_dict()
    for i in (1, 2, 3):
        dump(f"conv{i}_weight", sd[f"conv{i}.weight"].numpy())   # [out,in,kh,kw]
        dump(f"conv{i}_bias", sd[f"conv{i}.bias"].numpy())       # [out]
        dump(f"bn{i}_gamma", sd[f"bn{i}.weight"].numpy())
        dump(f"bn{i}_beta", sd[f"bn{i}.bias"].numpy())
        dump(f"bn{i}_mean", sd[f"bn{i}.running_mean"].numpy())
        dump(f"bn{i}_var", sd[f"bn{i}.running_var"].numpy())
    dump("fc_weight", sd["fc.weight"].numpy())  # [10,64]
    dump("fc_bias", sd["fc.bias"].numpy())      # [10]

    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")

    print(f"final test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
