# TODO

Status: `[ ]` open, `[~]` in progress, `[x]` done.

See `RESEARCH.md` for the findings behind these items (K200-on-modern-host work) and its "Open questions" section for the todo list this mirrors.

## Open

- [ ] Try pointing the official `paddlepaddle-xpu` wheel at this repo's own, proven-correct `libxpuapi.so` (via `LD_LIBRARY_PATH` or by replacing the wheel's bundled copy) to see if it fixes the silent-zero-result bug. ABI compatibility between the two builds is unknown; most promising untried lead for a working ML framework on this card. While there, also try PaddleNLP's XPU llama2-7b tutorial against K200 to get a hardware-confirmed answer on transformer-kernel support (see `RESEARCH.md` "Ollama / vLLM feasibility").
- [ ] `nm -D`/`c++filt` the Paddle wheel's own bundled `libxpuapi.so` (on the R730, where it's installed) for `xpu1::` RoPE/SwiGLU/RMSNorm/paged-attention symbols, to check whether it has LLM kernels this repo's bundled copy (confirmed to lack them) doesn't.
- [ ] Check whether Baidu's XTDK device compiler (needed to hand-write xpu1 cluster kernels for a custom llama.cpp/ggml backend) is publicly obtainable at all. Blocks any custom-backend attempt for Ollama support regardless of the kernel-availability finding above.
- [ ] Investigate the ~2x bandwidth shortfall vs the repo's documented peak (pageable H2D ~5.35 vs ~10.1 GB/s; `xpu_host_alloc` H2D ~6.29 vs ~12.5 GB/s) on the kernel-7.0 build, despite GEMM compute matching closely. Check PCIe link negotiation on this host/slot, NUMA placement, whether original S4-S9 numbers came from different hardware, or a regression in the stubbed DMA path.
- [ ] Test `rmmod`/reload of `kunlun.ko` and survival across a reboot when installed via `scripts/install_driver.sh`/DKMS (currently only loaded by hand).
- [ ] Test driver load under Secure Boot with a self-signed module (tested host currently has Secure Boot off).
- [ ] Determine whether any real, current driver/XRE version newer than this repo's exists via a direct Kunlunxin relationship (not publicly downloadable; may not be answerable from outside that channel).

## Done

- [x] Build `kunlun.ko`/`kunlun_peermem.ko` for kernel 6.15+/7.0 (gcc 14+): fixed `EXTRA_CFLAGS`→`ccflags-y` forwarding, added `-fms-extensions`/`-std=gnu11` to conftest probes, patched 6 genuine kernel API changes across 13 files, stubbed gcc-14-broken probes. Documented in `docs/BUILD-linux-7.md` and README's "Building on Linux 6.15 / 7.0".
- [x] Load the built driver on the R730 test host and confirm clean probe (`/dev/xpu0`, `/dev/xpu1`, `/dev/xpuctrl`, `/proc/xpu` reporting version `4.33.0`, no warnings/oopses).
- [x] Confirm the untested DMA stubs are actually correct (not just unbroken): `tests/test_host_alloc`, `tests/test_pageable_verify` pattern checks pass; `benchmarks/xpu_perf_test` GEMM FP16/FP32 and bandwidth results match the repo's documented baselines (bandwidth gap aside, tracked above). Updated `README.md`/`docs/BUILD-linux-7.md` language from "untested guesses" to "confirmed correct".
- [x] Isolate why the official `paddlepaddle-xpu` 2.6.1 wheel silently returns zeroed data: root-caused to the wheel bundling its own, different-build `libxpuapi.so` (not a kernel-module or XPURT transfer-layer bug).
- [x] Rule out current Baidu LLM-serving stacks (`vllm-kunlun`, FastDeploy, PaddleOCR/PaddleOCR-VL/PaddleX) for K100/K200 via upstream docs — all target Kunlun2/P800 or newer, none support K100/K200.
- [x] Search for a newer official K100/K200 driver/XRE release — none found publicly; this repo's patched `4.33.0` line appears to be close to the only version documented outside Baidu's internal/partner distribution.
- [x] Fix repo build-tooling gaps: `Makefile` hardcoded paths (`/usr/local/xpu-4.33.0`, `/mnt/storage/test_xpu`) replaced with self-contained defaults; missing `libxpurt.so`→`libxpurt.so.1` symlink now committed/auto-created; `git lfs pull` requirement called out in README Quickstart (was leaving `libxpuapi.so` and other large files as unfetched pointer files).
- [x] Document the `make clean`-at-repo-root hazard (deletes tracked conftest cache files under `kunlun-driver/kunlun_module`, recoverable via `git restore`) in `CLAUDE.md`.
- [x] Get a working Python 3.10 environment on a too-new host (Ubuntu 26.04, no 3.10 package) for the `cp310`-only `paddlepaddle-xpu` wheel, via pyenv-built 3.10.14.
</content>
