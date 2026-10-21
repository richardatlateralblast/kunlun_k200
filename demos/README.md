# Demos

Small, self-contained programs that trial a specific "could this card do X?"
idea from `RESEARCH.md`'s "Other workloads" / capability investigation, kept
around as working code instead of being written, run once, and deleted.

This is deliberately a third thing, distinct from:

- **`tests/`** — driver/DMA correctness regression (does the driver still move
  bytes correctly after a rebuild/reload). Part of `make regression`.
- **`benchmarks/`** — performance measurement against the documented baselines
  in `README.md`'s "Key benchmarks" table.
- **One-off scratch probes** — `CLAUDE.md`'s "standard procedure for testing a
  new op on real hardware" convention is to write a throwaway `.cpp`, run it,
  record the result in `RESEARCH.md`/`docs/research/`, then **delete** the
  scratch file. That's right for probing whether a single op dispatches at
  all. A demo here is the next step up: a complete, wired-together idea
  (several ops forming an actual small pipeline) that's worth keeping as
  runnable code rather than re-deriving from a doc's code block each time.

## What's here

Each demo is a toy-scale proof of concept for one capability already marked
confirmed-working in `README.md`'s Capabilities table — ported from the
reusable write-up in
[`docs/research/20261019-xpu1-op-testing-examples.md`](../docs/research/20261019-xpu1-op-testing-examples.md),
not a new, untested claim. A demo prints its result and a `PASS`/`FAIL`
sanity check against the hand-computed expectation from that write-up, same
pattern as `tests/`, so running it is also a quick "does this still work"
check, not just a read-the-output exercise.

| Demo | Capability | Source write-up |
|------|-----------|------------------|
| `kmeans_iteration` | K-means clustering, one full iteration (distance → assign → centroid update) | [#7](../docs/research/20261019-xpu1-op-testing-examples.md#7-k-means-one-iteration--confirmed-working-end-to-end) |
| `ncf_recommend` | NCF-style recommendation (embedding gather + dense MLP + sigmoid) | [#11](../docs/research/20261019-xpu1-op-testing-examples.md#11-ncf-style-recommendation-gather--fc--sigmoid--confirmed-working) |
| `video_pipeline_live` | Video/temporal modeling (CNN-per-frame → LSTM → classifier), fed **real** frames from an RTSP camera instead of synthetic data | [#12](../docs/research/20261019-xpu1-op-testing-examples.md#12-cnn-per-frame--temporal-lstm--dense-classifier-video-style-pipeline--confirmed-working), `RESEARCH.md` "Live RTSP feed" |
| `cifar10_live` | **Real recognition, not a mechanics demo**: a small CNN *trained on the real public CIFAR-10 dataset* (`scripts/train_cifar.py`), classifying real camera frames. 59.0% on-device accuracy against held-out CIFAR-10 test images (PyTorch reported 63.1% at training time). Weights in `cifar10_weights/`. | `RESEARCH.md` "Real recognition: CIFAR-10 classifier" |

All of these run at the same toy dimensions as their source write-up (a
handful of points/users/items) — scaling one up to realistic size is new,
untested work, not just "run the binary again."

## Adding a new demo

1. It should already be a confirmed-working result in `RESEARCH.md` and
   `README.md`'s Capabilities table — a demo documents and preserves a
   finding, it isn't where new hardware probing happens for the first time.
   Do that probing per `CLAUDE.md`'s standard procedure first, scratch file
   and all.
2. Keep the toy dimensions from the original probe unless you're
   deliberately taking on the (untested) work of scaling it up — say so in
   the demo's header comment and in this file's table if you do.
3. Print a `PASS`/`FAIL` line against the hand-computable expected result,
   and exit `0`/`1` accordingly, like `tests/` does.
4. Add a build rule in the root `Makefile` (`DEMOS` list) and a row in the
   table above, pointing back at the source write-up.
5. Needs the real card to run — building (`make demos`) only needs the
   bundled SDK, same as everything else in this repo.

## Live sources (cameras, etc.)

`video_pipeline_live` takes real input instead of baked-in toy data —
`scripts/run_video_pipeline_live.sh` pipes `ffmpeg`-decoded frames from an
RTSP URL into it. **Never put a camera's URL, hostname/IP, or credentials
into a file in this repo** — pass them at run time via the `RTSP_URL`
environment variable, per this repo's privacy convention (`CLAUDE.md`).

Public internet RTSP (port 554) is blocked outbound from the lab network
this repo's test host sits on — confirmed by direct TCP tests, not specific
to any one demo server (see `RESEARCH.md`'s "Live RTSP feed" section). A
camera on the same LAN as the test host works fine; a public stream won't.

## Training a real model (`cifar10_live`)

`scripts/train_cifar.py` trains the small CNN `cifar10_live` loads, against
the real public CIFAR-10 dataset (not a toy/synthetic one) — fetches a fast
mirror of it itself (the official `cs.toronto.edu` source was measured at
~500 bytes/s from this network; use a mirror, don't wait on that one):

```bash
curl -sL -o cifar10.tgz https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar xzf cifar10.tgz   # -> cifar10/{train,test}/<class>/*.png
python3 scripts/train_cifar.py --data-dir cifar10 --out demos/cifar10_weights --epochs 15
```

Needs `torch` and `Pillow` (CPU-only `torch` is fine — the model is tiny and
trains in minutes). The architecture is deliberately restricted to ops
already confirmed on xpu1 (plain conv2d, no grouped/depthwise; `batch_norm`
folded as fixed scale/bias at inference, see `cifar10_live.cpp`; no pooling
layers — downsampling via strided conv instead) so the trained weights are
guaranteed to run, not just plausible. `demos/cifar10_weights/` (~170KB) is
committed directly — retraining is only needed to change the architecture
or retrain from scratch, not to use the demo.
