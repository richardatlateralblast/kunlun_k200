# RESEARCH.md

Findings specific to running this exact K200 card and driver on a modern host (kernel 6.15+, including 7.0; gcc 14+), and to what software can actually use it today. Kept separate from `README.md`, which documents the repo's original S0–S9 driver-tuning work and benchmark baselines on kernel 6.8. Start here before new work on this card; it records what's confirmed, what's broken, and what's genuinely open, so it isn't silently redone or contradicted.

Dates below are when each finding was made, not when the underlying software shipped.

## Hardware and software generations (context for everything below)

Kunlunxin's cards are not one driver line with version numbers — each silicon generation has its own driver and, increasingly, its own software ecosystem:

| Generation | Chip | Cards | Driver/SDK line | Current official software support |
|---|---|---|---|---|
| Kunlun1 (KL1) | — | **K100, K200** (this repo) | `4.33.0` (this repo, patched for kernel 7.0), older `1.1.0` package line also seen in the wild | Mainline Paddle versions after 2.3 reportedly dropped it (unconfirmed, see below); no vLLM; FastDeploy doesn't list it |
| Kunlun2 | — | R200, R300 | separate, newer | Current PaddleOCR/PaddleX XPU docs target this and P800 |
| Kunlun3 (KL3) | — | **P800** | newest, `xpytorch` (KL3-customized PyTorch) | Target of `vllm-kunlun`, current `paddlepaddle-xpu`/FastDeploy releases |

Nothing found in this investigation contradicts the conclusion that **K100/K200 is two full hardware generations behind what Baidu's current LLM-serving software targets**. Treat any official Kunlunxin software released from 2026 onward as written for P800 unless it explicitly says otherwise.

## Driver: building for kernel 6.15+/7.0

Full step-by-step instructions are in `docs/BUILD-linux-7.md` and the README's "Building on Linux 6.15 / 7.0" section; this is the summary of *why* each change was needed, found by building on a Dell R730 (kernel `7.0.0-38-generic`, gcc 15.2) from a stock clone of this repo.

- **Build-system breakage, not kernel-API breakage, caused most of the ~225 initial compile errors.** Two separate causes:
  - Kernel 6.15+ ignores `EXTRA_CFLAGS`, and this driver's `Kbuild` puts every include path and `-D` define there. Fix: forward it to `ccflags-y`.
  - The conftest feature-detection probes compile without `-fms-extensions`, which kernel 7.0's headers need; every probe that pulls in `pci.h` or `scatterlist.h` then fails on `static assertion failed: sizeof(struct filename) % 64 == 0`, and the driver silently takes 6.8-era fallback code (`mmap_sem`, `vm_fault.virtual_address`, 8-argument `get_user_pages`). Fix: add `-std=gnu11 -fms-extensions` to the conftest's own `NV_CONFTEST_CFLAGS`, found by redirecting the conftest's `$CC ... 2>&1` (normally discarded) to a log file.
  - A secondary issue: the "function is present" conftest probes are logically inverted (success = absent) and rely on an *implicit declaration* being only a warning. gcc 14+ makes that a hard error by default, so several functions (`phys_to_dma`, `dma_is_direct`, `acquire_console_sem`, `jiffies_to_timespec`, `unsafe_follow_pfn`) were wrongly detected as present. Needed `-std=gnu11` (gcc 15 defaults to C23, which also breaks this) and `-Wno-error=int-conversion`.
  - gcc 13 is not a workaround: it passes the probes, but kernel 7.0's own build uses `-fmin-function-alignment`, which gcc 13 rejects.
- **Six genuine kernel API changes**, patched in source (13 files, ~30 lines): `in_irq()`→`in_hardirq()`; `del_timer()`→`timer_delete()` (6.15); `hrtimer_init()`+`.function=` →`hrtimer_setup()` (6.15); `nth_page(p,j)`→`p+j` (6.17, safe here because the pages come from one contiguous allocation); `follow_pfn()`/`unsafe_follow_pfn()`→`follow_pfnmap_start/end()` (6.12); `dma_map_ops.map_resource` removed (peer-DMA capability check now returns false).
- **Stubs for the gcc-14-broken probes above:** `phys_to_dma` falls back to the plain physical address; `dma_is_direct` assumed false; the console-lock and `jiffies_to_timespec` branches disabled; `vm_flags_set`/`vm_flags_clear` always used (`vm_flags` has been const since 6.3). These were written as *untested guesses* and flagged as a correctness risk — **since confirmed correct**, see "Confirmed working" below.
- **Result:** `make modules CC=<gcc -std=gnu11 wrapper>` → `kunlun.ko` (vermagic matches `uname -r`, no unresolved symbols), `kunlun_peermem.ko`.

## Driver: loading (2026-10-18, R730 test host)

Checked before loading: Secure Boot disabled, kernel lockdown `none`, host idle. Loaded by hand (`sudo insmod kunlun.ko kl1_p2p_stub=0 kl1_dma_direct=1 kl1_bounce_pipe=1`), not through `scripts/install_driver.sh`.

- Exit 0, ~7s to probe. `dmesg`: `module verification failed: signature and/or required key missing — tainting kernel` (expected, unsigned module), `load kunlun driver 4.33.0`, clean probe of `0000:82:00.0` (BARs 64MB/64KB/2MB, MSI set up), `module load success`. No warnings, oopses, or call traces.
- `/dev/xpu0`, `/dev/xpu1`, `/dev/xpuctrl` appeared (mode 666, major 234); `/proc/xpu` reports version `4.33.0`, `probe_error` empty, `wait_mode` hybrid.
- **Not yet tried:** unload/reload, persistence across a reboot (loaded by hand, not installed via the install script/DKMS path), running with Secure Boot on (would need signing).

## Confirmed working: driver DMA and xdnn compute are correct (2026-10-18)

An earlier hypothesis — that the untested DMA stubs above were broken — was **wrong**, and the actual bug was found and isolated:

1. **Isolated the failure to transfer, not arithmetic.** A pure round-trip through Paddle (`paddle.to_tensor(host_array).numpy()`, no math at all) came back zeroed, same as `1+1` returning `0`. This ruled out "wrong compute, right transfer."
2. **Found and fixed a real repo gap along the way:** `xdnn-ubuntu_x86_64/so/libxpuapi.so` was a Git LFS pointer (134 bytes), never fetched, in *every* clone checked (the R730 test clone and the separate local clone this `RESEARCH.md`/`CLAUDE.md` work was done from). `git lfs install && git lfs pull` fixed it (~650MB real file). Now fixed at the repo level — see "Repo/build fixes" below.
3. **Built this repo's own correctness tests** (`tests/test_host_alloc`, `tests/test_pageable_verify`) against the real XPURT library. **Both passed their pattern checks.**
4. **Ran this repo's own `benchmarks/xpu_perf_test` (full GEMM + bandwidth suite) to completion.** (It only *looked* hung under a short timeout — its stdout wasn't flushed; given the time it needs — about 1–2 minutes — it finishes cleanly, exit 0.) Results:
   - GEMM FP16: 19,151–22,597 GFLOPS at 32.96–39.23% efficiency (README's own documented baseline: ~20 TFLOPS, ~35%).
   - GEMM FP32 "large" (2048³): 7685 GFLOPS at 53.37% (documented: ~7.8 TFLOPS, ~54%).
   - INT8: correctly reports "not supported" (matches documented KL1 firmware limitation).
   - Bandwidth: real and non-zero, but **roughly half the documented peak** — pageable H2D up to 5.35 GB/s (documented ~10.1), `xpu_host_alloc` H2D up to 6.29 GB/s (documented ~12.5). This is a performance gap, not a correctness bug — flagged as open below.

**Conclusion: on this patched kernel-7.0 driver, the DMA stubs are correct and direct xdnn/XPURT compute matches the repo's own documented numbers.** The `README.md` and `docs/BUILD-linux-7.md` "Stubs" language has been updated from "untested guesses" to "confirmed correct" accordingly.

## Confirmed broken: the official `paddlepaddle-xpu` wheel (2026-10-18)

With the driver above loaded and working (confirmed by the tests in the previous section), the official pip wheel still fails at the compute level:

- Installed via `pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl` (needs Python 3.10 — see "Getting Python 3.10" below; no official index URL found for 2.6.1, only this direct wheel link, via PaddleX's Kunlun XPU docs).
- `paddle.is_compiled_with_xpu()` → True, `paddle.device.xpu.device_count()` → 2 (matches the two `/dev/xpu*` nodes), `paddle.set_device("xpu:0")` succeeds. Detection works fine.
- But: `paddle.to_tensor(host_array).numpy()` returns all zeros (silent data loss, no exception); elementwise add and matmul of all-ones tensors also return all zeros; `paddle.randn(...)` raises `OSError: ... XDNN Error, XDNN_INVALID_PARAM`.
- **Root cause, isolated:** Paddle's wheel bundles its own `libxpuapi.so` (587MB, version string `XDNN_VERSION_master-changed`, build workspace paths reference newer kernel source such as `xpu/refactor/page_attention.h` that doesn't exist in this repo's own bundled copy). This repo's bundled `libxpuapi.so` is a *different build* (650MB) and is proven correct (previous section). The kernel module and XPURT transfer layer are not the cause — both are shared/identical between the two scenarios and both check out correct.
- **Corroborating public report:** [PaddlePaddle/Paddle#56781](https://github.com/PaddlePaddle/Paddle/issues/56781) — a different user reported Paddle release/2.5 "compiled but hung at runtime" on K200 (different symptom, same underlying story: Baidu's official Paddle builds don't reliably run compute correctly on K200, even when they load and detect the device). A community reply in that thread claims Paddle versions after 2.3 dropped K200 support outright; this is **unconfirmed** — no official release note was found, and the thread's actual evidence is messier (a 2.2 build failed only because Baidu's CDN URLs for the XRE/XDNN tarballs had expired, an infrastructure problem, not a version gate).
- **Not yet tried:** forcing Paddle to load this repo's own (proven-correct) `libxpuapi.so` instead of its bundled one, e.g. via `LD_LIBRARY_PATH` preload. Worth trying given the raw library is proven to work; ABI compatibility between the two builds is unknown.

## Ruled out: current LLM-serving software

Checked via upstream docs/repos, not by installing (install would be pointless given what's documented):

- **`baidu/vllm-kunlun`** (vLLM hardware plugin): its own FAQ states only the Kunlun3 (P800) series is currently supported; Kunlun2 (R200) and Kunlun4 (M100/M300) are explicitly listed as *not yet* supported, with M100 support "planned for early 2026." K200 isn't named anywhere in the repo. It requires Python 3.10+, PyTorch 2.5.1+ via `xpytorch` (a **KL3-customized** PyTorch build — i.e. this isn't a driver-version gap, it's a different runtime stack that doesn't exist for KL1). An open feature request, [baidu/vLLM-Kunlun#499](https://github.com/baidu/vLLM-Kunlun/issues/499) (filed 2026-10-12), asks for K200 support and reports the card "can't currently run even small models" through it; zero maintainer response as of this writing.
- **FastDeploy** (Baidu's LLM deployment toolkit, built on Paddle): its Kunlunxin XPU install guide requires **driver ≥ 5.0.21.26** and firmware ≥ 1.48, verified only on the P800-OAM board. Our driver reports `4.33.0` — a different, older line, and no official source names a 5.x driver for K100/K200 at all (see "Driver versioning" below). `paddlepaddle-xpu` 3.x (which FastDeploy needs) is pinned to the `xpu-p800` package index.
- **PaddleOCR / PaddleOCR-VL / PaddleX** current docs: list Kunlun R200/R300/P800 as supported XPU targets. K100/K200 isn't named in any of them. PaddleOCR-VL's own Kunlunxin tutorial says it's verified only on P800.

## Driver versioning: no newer official release found

Searched specifically for whether a newer driver exists for K100/K200 (independent of the P800-only projects above):

- No official Kunlunxin changelog, release notes, or download page was found for K100/K200 driver or XRE SDK versions at all — not a comparison table, just nothing publicly indexed.
- A different, older version line (`kunlun-xre_1.1.0_linux.run`, `kunlun-driver_1.1.0_amd64.deb`) surfaced in an unrelated PaddlePaddle GitHub issue ([#79055](https://github.com/PaddlePaddle/Paddle/issues/79055)), where the reporter says those exact packages are **no longer downloadable from the web or from Baidu's own BOS object-storage archive** — i.e. even Baidu's own historical mirror doesn't have an old version anymore, which is a bad sign for finding a newer one as an outsider.
- One secondary source (a PP-OCR deployment write-up) puts the matching driver floor for K200 at "3.8 and up" — nowhere near FastDeploy's `5.0.21.26` requirement, and inconsistent with the `4.33.0`/`1.1.0` version strings seen elsewhere, suggesting there may be more than one numbering scheme in circulation for this hardware.
- **Conclusion:** there is no publicly fetchable "updated K200 driver." The driver in this repo (patched from `4.33.0`) is, as far as open sources show, close to the only version documented anywhere outside Baidu's internal/partner distribution.

## Repo/build fixes made during this work

Separate from the driver source patches above — these are gaps in the repo's own build tooling, now fixed and committed:

- **`Makefile` hardcoded paths that don't exist on a fresh clone**: `-I/usr/local/xpu-4.33.0/include`/`-L/usr/local/xpu-4.33.0/lib64` (an external XPURT SDK install) and `ROOT = /mnt/storage/test_xpu` (a specific developer's machine). Fixed: `ROOT` defaults to the checkout itself; `xpu/runtime.h` (the low-level XPURT header) is found under this repo's own `kunlun-driver/include`, previously unreferenced by the Makefile; `XPURT_INC`/`XDNN_INC`/`XDNN_LIB` are overridable for anyone who does have a system SDK install.
- **The `libxpurt.so` → `libxpurt.so.1` symlink that `-lxpurt` needs** wasn't present (the SDK only ships the versioned `.so.1`) and wasn't LFS-tracked content, so no clone had it. Now both a Makefile prerequisite (auto-created) and committed directly.
- **`git lfs pull` was never run** in any clone checked, leaving `libxpuapi.so` (and a few other large files: `data/xpu_denoise_synth.bin`, `paddle_models/inference.pdiparams`, `xdnn-ubuntu_x86_64/lib/libxpuapi_ut.a`) as unfetched pointer files. No code fix possible for this — it's a one-time `git lfs install && git lfs pull` after cloning, now called out in the README's Quickstart.
- **Hazard found, not a bug to fix:** running `make clean` at the repo root recurses into `kunlun-driver/kunlun_module` and deletes files that are tracked in git (the conftest compile-test cache). This happened twice during this work; both times recovered losslessly with `git restore kunlun-driver/` since the files are already committed. Documented in `CLAUDE.md` as a thing to avoid, not changed in the Makefile itself (out of scope here — it's pre-existing repo structure, not something introduced by the kernel-7.0 work).

## Getting Python 3.10 on a too-new host

`paddlepaddle-xpu` 2.6.1 ships only a `cp310` wheel. The R730 test host runs Ubuntu 26.04 ("resolute"), whose repos carry Python 3.12 (library only, no `python3.12` binary package) and 3.14, but no 3.10 at all.

- First attempt: compiling CPython 3.10.14 from source by hand into a user prefix. Works, but it's a one-off with no record of how it was built.
- **Used instead, at the user's request: pyenv.** `git clone https://github.com/pyenv/pyenv.git ~/.pyenv`, then `PYENV_ROOT=~/.pyenv PATH=$PYENV_ROOT/bin:$PATH pyenv install 3.10.14` (needs `build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncurses-dev uuid-dev tk-dev wget` from apt first). Takes the same ~15–20 minutes to compile either way (pyenv is a wrapper around the same `./configure && make`), but gives a reproducible, documented install instead of a hand-rolled one.
- Then: `~/.pyenv/versions/3.10.14/bin/python3.10 -m venv <venv>`, `pip install <paddlepaddle-xpu wheel URL>`.
- **Caution on background builds:** a `make -j64` kicked off via `ssh ... & disown` returns control to the caller immediately (the `ssh` wrapper exits once it has launched and detached the job) — a task-completion notification for that `ssh` call does **not** mean the actual build finished. Verify by checking for the real process (`pgrep -f`) or a completion marker file on the remote host before trusting that something is done.

## Ollama / vLLM feasibility: static analysis (2026-10-18, no hardware access — machine off)

Investigated whether either is realistically achievable on K200, without assuming "no official support" is the whole story.

**Confirmed via web search: neither has any existing path.**

- vLLM: Baidu's own `vllm-kunlun` plugin supports Kunlun3 (P800) only and runs on `xpytorch`, a KL3-customized PyTorch build distributed as a `.run` installer; nothing comparable exists for KL1. (Already established pre-2026-10-18, re-confirmed here.)
- Ollama: it runs on `llama.cpp`/`ggml`. No Kunlun backend of any kind was found in ggml, upstream or third-party — `llama.cpp`'s backend list (CUDA, HIP, Metal, Vulkan, MUSA, SYCL) doesn't mention Kunlun, and no PR/issue proposing one turned up. Baidu's own Kunlun investment is in Paddle and vLLM, not llama.cpp.
- PaddleNLP (Baidu's own LLM-serving path, the closest thing to "does any transformer model run on Kunlun hardware via Baidu's own tooling") targets **Kunlun2 (R300/R480, driver 4.31.0)** for its documented llama2-7b tutorial — not K100/K200. This matches the finding below.

**New finding, from inspecting this repo's own `libxpuapi.so` directly (no card needed — `nm -D`/`strings` on the 650MB file already in this checkout):**

The library's compute kernels are namespaced by hardware generation: `xpu2::` (~15,700 functions), `xpu3::` (~8,000), `xpu4::` (~245) — presumably Kunlun2/3/4. **Zero symbols exist in an `xpu1::` namespace** (K100/K200's architecture). Every transformer-serving primitive checked — `rotary_pos_embedding`/RoPE, `swiglu_fwd`/`swiglu_bwd`, `rms_layer_norm_infer` (RMSNorm), `qkv_paged_attention` — exists only under `xpu2::`/`xpu3::`.

K100/K200 (xpu1) code does exist in the same library, but only for older-generation ops, as flat (non-namespaced) C functions: GEMM/FC/conv (matches this repo's own working GEMM benchmarks) and `XPU1ActTable`, whose lookup tables cover `relu`/`sigmoid`/`tanh`/`gelu`/`swish`/`hard_sigmoid`/`hard_swish` — i.e. CNN- and BERT-era activations, with **no SwiGLU and no RMSNorm table**. This lines up with what K200 was actually built and validated for in this repo (ResNet-50, BERT-style fixed-size attention), not LLaMA-family decoder architectures.

**Reading:** this isn't a driver-version or packaging gap like the `paddlepaddle-xpu` silent-zero bug (open question above) — the low-level kernels current LLM serving needs appear to have never been written for KL1 silicon in Baidu's xdnn library, at least not in the build bundled in this repo. Not yet checked: whether the *Paddle wheel's own* bundled `libxpuapi.so` (587MB, a different build) has any `xpu1::` LLM kernels this repo's copy lacks — a cheap `nm -D` check, but it needs the R730 host where that wheel is installed, not this machine.

**Plan of attack, ranked by cost and what each step would actually prove:**

1. **Cheapest next step, on the R730 (combine with the already-planned `LD_LIBRARY_PATH` fix above):** once the `paddlepaddle-xpu` wheel is pointed at this repo's proven-correct `libxpuapi.so`, try running PaddleNLP's own XPU llama2-7b tutorial against K200 and capture the real failure. This turns the binary-analysis hypothesis above into a hardware-confirmed fact (or disproves it) for near-zero extra cost, since the environment would already be set up for the wheel fix.
2. **Cheap, no hardware needed:** `nm -D`/`c++filt` the Paddle wheel's own bundled `libxpuapi.so` for `xpu1::` RoPE/SwiGLU/RMSNorm/paged-attention symbols — confirms whether this repo's copy is simply missing kernels a newer build has, before concluding they're absent for KL1 everywhere.
3. **If (1)/(2) confirm the kernels genuinely don't exist for xpu1:** a custom `llama.cpp`/ggml backend calling into xdnn directly is *conceivable* (xdnn's kernels are written against the `xpu/kernel/cluster*.h` SIMD/cluster-primitive headers bundled in this repo, so the programming model is visible), but would mean hand-writing RMSNorm/RoPE/SwiGLU/attention device kernels from scratch against those primitives and compiling them with Baidu's XTDK device compiler — which is **not in this repo and not confirmed to be publicly obtainable**. Check XTDK availability before investing further; if it isn't realistically obtainable outside a Baidu partnership, this path is closed, not just hard. Even in the best case, K200's firmware has no INT8 CDNN kernel (already confirmed above), so the result would be FP16-only — no GGUF-style quantization, limiting it to small models fitting 8GB HBM/XPU in FP16.
4. **vLLM: not worth independent effort.** It would need a from-scratch equivalent of `xpytorch` (full PyTorch device backend: ATen op coverage, autograd, etc.) sitting on top of the same missing kernel layer found in (3) — strictly more work than (3) for the same underlying gap, with no indication Baidu plans to ship `xpytorch` for KL1.

## Other workloads this card's actual kernel set fits (2026-10-18, same static analysis)

While checking `libxpuapi.so` for LLM kernels (previous section), also checked what *is* present for xpu1 (K100/K200), beyond the GEMM/conv/activation ops already confirmed by this repo's own benchmarks. Ranked by how directly the evidence supports each:

- **K-means clustering — best-supported idea found so far; no unconfirmed ops needed.** The whole algorithm maps onto primitives that are flat/non-namespaced (so xpu1-available) with nothing gated to `xpu2::`/`xpu3::`: distance-to-centroids is `‖x‖² + ‖c‖² - 2x·cᵀ` (GEMM, already proven, plus `square_reduce_sum`); assignment is `argmin` over the distance row (present as both a flat C function and the templated API); and the centroid update — normally a group-by-sum, the same shape that blocks GNN message passing — sidesteps that entirely via the standard trick of building a one-hot assignment matrix (`one_hot`/`one_hot_fwd`, present, non-namespaced) and computing per-cluster sums as a single GEMM (`one-hotᵀ @ X`) with counts via `reduce_sum`. Every op in that chain is confirmed present and ungated, unlike the CTR/ranking or CNN ideas below which still need a real test to confirm end-to-end behavior. Caveat: it's an iterative distance→assign→update loop, several kernel launches per pass, fine for batch clustering, not for sub-millisecond online use.
- **Recommendation / ranking / CTR-style models — looks like this card's original intended workload.** Non-namespaced (so presumably xpu1-available) symbols include `search_grnn`/`search_bi_grnn`/`search_grnn_pointer_max` (Baidu's GRNN cell, used in its search/ads ranking stack), `search_seq_softmax`/`multi_softmax`, embedding `gather`/`gather_nd`/`scatter_assign`, and `baidu::xpu::api::random<T>` (on-device Philox-backed RNG, useful for dropout/init/sampling). Together these are the building blocks of a classic GRNN/MLP/embedding CTR or learning-to-rank model (DSSM/Wide&Deep/DIN-style), which predates LLaMA-era transformers and matches when this chip generation shipped. Untested on real weights, but every needed primitive appears present — the most promising *new* workload to actually try.
- **Classic CNN vision, beyond ResNet-50/denoise already proven.** `batchnorm_fwd_infer`/`_train`/`_bwd`, `pooling_fwd`, `avg_pooling_backward`, `max_pooling_backward`, and `conv2d_fusion` all present without generation-specific namespacing. Object detection or segmentation models from the same CNN-backbone era (YOLO-v3/v4-class, U-Net-class) should fit the same envelope as the already-working ResNet-50/denoise cases. `fusion_cv.h` also exposes fused `resnet_unit_fusion` (ResNet-34/50/101/v10) *and* **`yolo_unit_fusion`** kernels specifically — object detection backbones, not just classification.
  - **OpenCV: no native path, but a standard hybrid pipeline works.** OpenCV has no Kunlun/XPU backend, and its one vendor-agnostic GPU route (the T-API/`UMat` OpenCL path) needs an OpenCL ICD this card doesn't have — checked `libxpurt.so.1` for `cl*` entry points, none exist. There's also no on-device image-decode/resize in this SDK (no JPEG/codec headers under `xdnn-ubuntu_x86_64`), unlike some vendor SDKs. The realistic pattern is the usual GPU-inference split: OpenCV on the CPU for capture/decode/resize/color-convert/drawing, `xpu_memcpy` the preprocessed tensor across, run conv/pooling/batchnorm/`yolo_unit_fusion` inference on the card via xdnn, copy the result back for OpenCV to draw. Nothing in this repo uses OpenCV today (not referenced anywhere, not installed on this dev machine) — this would be new integration work, but the same shape as the existing `benchmarks/`/`tests/` code (direct `xpu_memcpy` + xdnn calls), not a new kind of problem.
- **Classic (non-VL) OCR inference — conv+RNN+softmax backbone looks present, training may not.** CRNN-style detection/recognition (conv backbone + GRNN/LSTM + softmax decode) uses only ops confirmed above. Caveat: `ctc_loss_bwd`/`ctc_cost_grad` (CTC-loss *training*) exist only under `xpu2::`/`xpu3::` — so training a new CTC model here is likely blocked the same way LLM training is, but running a *pretrained* CRNN for inference (greedy/beam decode on the softmax output, no CTC-loss kernel needed) looks feasible.
- **BERT/ERNIE-era encoder models (pre-LLaMA transformers) — plausible but unconfirmed.** The plain (non-RMS) `baidu::xpu::api::layer_norm<T>` wrapper and `XPU1ActTable::get_gelu_table` both exist without generation gating, and GEMM+softmax (both confirmed working) are enough to hand-compose unfused multi-head attention the way pre-2023 model code did, instead of calling the xpu2/3-only fused `attention.h`/`page_attention.h` kernels. Nothing confirms this runs end to end — worth a real test with a small BERT/ERNIE checkpoint before assuming it works.
- **Graph neural networks — weak fit, likely blocked.** GNN message passing needs scatter-add/segment-sum (aggregating a variable number of neighbor features per node) or sparse matmul (SpMM) as its core primitive. `index_add` (the actual scatter-add kernel) exists only under `xpu2::`/`xpu3::`, same generation-gating as the LLM ops, and there's no `segment_sum`/SpMM/CSR support anywhere in this xdnn SDK regardless of generation. What's ungated — generic `gather`/`gather_nd`/`scatter_assign` (`assign`, not `add`, so no built-in accumulation), and `sparse_adam`/`sparse_sgd` (sparse-gradient optimizers for embedding tables, not graph aggregation) — isn't enough to build real message passing on. Baidu's own GNN framework, PGL, is built on Paddle but nothing in its docs claims Kunlun-kernel support, and it would need the same missing scatter-add primitive. A GNN layer here would mean hand-writing a scatter-add/segment-sum kernel against the raw cluster-primitive headers — the same XTDK-availability blocker as the custom ggml-backend path above — not just wiring up an existing op.
- **Bare compute/RNG coprocessor, bypassing any ML framework entirely.** This is what this repo's own benchmarks and tests already do (direct xdnn/XPURT GEMM calls), and it's the lowest-risk way to use the card for anything: batched FP16/FP32 matrix math, image/signal processing (the existing denoise case), or Monte Carlo-style workloads using the on-device RNG — no dependency on any serving stack's device-generation gating at all.

## Open questions

- Does pointing the official `paddlepaddle-xpu` wheel at this repo's own (proven-correct) `libxpuapi.so`, via `LD_LIBRARY_PATH` or by replacing the bundled file, fix the silent-zero-result bug? ABI compatibility between the two builds is unknown. This is the most promising untried lead for getting a standard ML framework working on this card. **Next session: also try PaddleNLP's XPU llama2-7b tutorial on top of the fix — see "Ollama / vLLM feasibility" above.**
- Does the Paddle wheel's own bundled `libxpuapi.so` (587MB) contain `xpu1::` RoPE/SwiGLU/RMSNorm/paged-attention kernels that this repo's bundled copy (650MB) lacks? Needs the R730 host. See "Ollama / vLLM feasibility" above.
- Is Baidu's XTDK device compiler (needed to hand-write xpu1 cluster kernels) publicly obtainable at all, independent of a partner relationship? Blocks any custom llama.cpp/ggml-backend attempt regardless of the kernel-availability finding above.
- Why is bandwidth roughly half the repo's originally-documented peak (pageable H2D ~5.35 vs ~10.1 GB/s; `xpu_host_alloc` H2D ~6.29 vs ~12.5 GB/s) on the kernel-7.0 build, when GEMM compute matches almost exactly? Candidates not yet checked: PCIe link negotiation differences on this specific host/slot, NUMA placement, whether the original S4–S9 numbers were tuned on different hardware, or an actual (if non-corrupting) regression in the stubbed DMA path.
- Does the module survive `rmmod`/reload, and does it survive a reboot when installed via `scripts/install_driver.sh`/DKMS rather than loaded by hand? Not tried.
- Does this driver load and work under Secure Boot with a self-signed module? Not tried (tested host has Secure Boot off).
- Is there a real, current driver or XRE version newer than what's in this repo, obtainable only through a direct relationship with Kunlunxin (not public downloads)? Not something answerable from outside that channel.
