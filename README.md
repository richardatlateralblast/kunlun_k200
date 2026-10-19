# Kunlun K200 XPU Performance Analysis & Driver Tuning

[中文版 / Chinese](README_cn.md)

Benchmarks and KL1 driver work for **Kunlun K200** inference cards: PCIe bandwidth, GEMM, Paddle ResNet-50, real-case demos (C1–C5), and driver fixes for same-card P2P / `xpu_host_alloc`.  
Same host also tracks **Intel Xeon Phi 7220P (KNL)** bring-up and co-located AMD MI50 GPUs (not part of the K200 driver milestone).

**Status (2026-08, kernel 7.0 row updated 2026-10):**

| Track | State |
|-------|--------|
| K200 driver S0–S9 | **Closed** → [`docs/impl/20260715-project-closure.md`](docs/impl/20260715-project-closure.md) |
| K200 cases C1–C5 + demo | **Delivered** → [`results/cases/SUMMARY.md`](results/cases/SUMMARY.md) |
| K200 next work | Productize / real models / freeze → [`docs/plan/20260716-phase-after-abc.md`](docs/plan/20260716-phase-after-abc.md) |
| K200 on kernel 7.0 / modern host | Driver builds, loads, compute confirmed correct; official Paddle wheel still broken → [Kernel 7.0 / modern host support](#kernel-70--modern-host-support), [`RESEARCH.md`](RESEARCH.md) |
| Phi 7220P (KNL) | Phase 0 done · Phase 1 blocked on **Rocky 8 + MPSS 4.4.1** → [`docs/plan/20260805-xeon-phi-7220p-bringup.md`](docs/plan/20260805-xeon-phi-7220p-bringup.md) |

**Repo:** [github.com/mobius/kunlun_k200](https://github.com/mobius/kunlun_k200) (`master`)

**Privacy:** Public docs must not include hostnames, absolute home/lab paths, full `lspci`/dmesg dumps, BMC details, or device serial numbers. Use PCI **IDs** and relative repo paths only.

### Quickstart

```bash
git clone <this repo> && cd kunlun_k200
git lfs install && git lfs pull   # fetches libxpuapi.so (~650 MB) and other large files
make all                          # builds benchmarks/ and tests/ against the bundled xdnn SDK — no external install needed
./tests/test_host_alloc && ./tests/test_pageable_verify   # correctness check
./benchmarks/xpu_perf_test 0      # full GEMM + bandwidth suite
```

This assumes the kernel driver is already loaded (`lsmod | grep kunlun`). If not, or if you're on kernel 6.15+ (including 7.0), build and load it first — see [Install](#install) and [Kernel 7.0 / modern host support](#kernel-70--modern-host-support). **The built `kunlun.ko` is never committed** (`*.ko` is gitignored, like every compiled binary in this repo); there's no prebuilt module to download, so this build step is mandatory, not optional, for every kernel version including the ones this driver originally targeted.

### Live driver check (K200)

```bash
modinfo -F srcversion,vermagic kunlun
uname -r    # vermagic must match
cat /sys/module/kunlun/parameters/kl1_dma_direct \
    /sys/module/kunlun/parameters/kl1_bounce_pipe \
    /sys/module/kunlun/parameters/kl1_pageable_pin \
    /sys/module/kunlun/parameters/kl1_p2p_stub
```

| Item | Production expectation |
|------|------------------------|
| Knobs | `kl1_dma_direct=1`, `kl1_bounce_pipe=1`, `kl1_pageable_pin=0`, `kl1_p2p_stub=0` |
| `vermagic` | Equals `uname -r` (rebuild after kernel upgrade) |
| `srcversion` | Changes every rebuild — compare to `modinfo -F srcversion kunlun-driver/kunlun.ko`, not a fixed string in docs |

After rebuild, record your own `srcversion` next to the knobs above. Older case reports may cite a previous fingerprint; that string is not a permanent ID.

---

## Hardware

| Component | Detail |
|-----------|--------|
| K200 | Baidu Kunlun AI Accelerator `1d22:3684` (KL1), **2 PD / card** → `/dev/xpu*` |
| Inventory (lab) | Multiple K200 cards (typically 2–4); optional co-located AMD MI50 and Intel Phi — not all targets of this repo |
| Arch / clock | KL1, 900 MHz locked |
| HBM | 8 GB per XPU (16 GB per card) |
| PCIe (K200) | Gen4 ×8 per card |
| Phi 7220P (side track) | Knights Landing, PCI ID `8086:2260` (+ DMA fn `8086:2264`); **no driver yet** |
| Host (generic) | x86_64 server-class CPU, Ubuntu 22.04, kernel **6.8.x**, or kernel **7.0** with gcc 14+ (see [Kernel 7.0 / modern host support](#kernel-70--modern-host-support)); rebuild `kunlun.ko` after kernel upgrade |

> Do **not** publish hostnames, absolute paths, full `lspci` dumps, or device serials in docs. Discover local BDFs with `lspci -nn \| grep -E '3684\|2260'`.  
> Case numbers were measured on one XPU / same-card PD pairs; multi-card count does not change S4–S6 bandwidth floors.

### Phi 7220P (side track)

**7220P = Knights Landing → MPSS 4**, not MPSS 3 / 7120P (KNC). Do not reuse `-mmic` / `micnativeloadex`.  
Override BDF via env (`PHI_BDF_MAIN` / `PHI_BDF_AUX`) if your slot differs from script defaults.

```bash
bash scripts/check_phi_7220p.sh 2>&1 | tee /tmp/phi_7220p_hw.log   # keep logs local; do not commit raw dumps
bash scripts/phase1_mpss4_readiness.sh
python3 scripts/phi_discover.py
```

Plan / research: [`docs/plan/20260805-xeon-phi-7220p-bringup.md`](docs/plan/20260805-xeon-phi-7220p-bringup.md), [`docs/research/20260805-phi-7220p-phase1-mpss4.md`](docs/research/20260805-phi-7220p-phase1-mpss4.md).  
Large vendor trees under `third_party/` are **gitignored**.

---

## Capabilities (K200 / KL1)

Single at-a-glance index of what's actually confirmed on this card, kept up to date as findings land (see maintenance note below). Detail and evidence for every row live in [`RESEARCH.md`](RESEARCH.md) and the sections linked inline; this table is the summary, not the write-up.

Legend: ✅ confirmed working · ❌ confirmed broken · 🚫 confirmed unsupported (hardware/software gap, not fixable by config) · ⚠️ plausible, not yet tested end-to-end · ❓ untested, no evidence either way

### Driver & platform

| Capability | Status | Notes |
|---|:---:|---|
| Build for kernel 6.15+/7.0 (gcc 14+) | ✅ | `kunlun.ko`/`kunlun_peermem.ko` build clean — [Kernel 7.0 / modern host support](#kernel-70--modern-host-support) |
| Driver load / probe | ✅ | Clean probe, `/dev/xpu0`/`xpu1`/`xpuctrl`, no warnings/oopses |
| DMA correctness (`xpu_host_alloc`, pageable) | ✅ | `tests/test_host_alloc`/`test_pageable_verify` pattern checks pass |
| GEMM compute, FP16/FP32 | ✅ | Matches this repo's own documented baselines — [Key benchmarks](#key-benchmarks-k200-post-s9--cases) |
| GEMM compute, INT8 | 🚫 | Correctly reports unsupported — KL1 firmware limitation, not a driver bug |
| PCIe bandwidth at documented peak | 🚫 | **Root-caused**: this test host's CPU (Broadwell-EP) has no PCIe Gen4 controller — link negotiates to Gen3 (8GT/s) vs the card's rated 16GT/s, confirmed via `lspci` (`LnkSta: 8GT/s (downgraded)`). A hard platform limit, not a driver/software issue — would need different (Gen4-capable) host hardware. See `RESEARCH.md` |
| Reboot persistence (either install script) | ✅ | Confirmed via a real reboot on the R730 host, both the manual and real-DKMS scripts |
| Kernel-*upgrade* persistence (`install_driver_dkms.sh`) | ✅ | Confirmed end to end: installing a new kernel package auto-triggers `dkms_autoinstaller`, which builds/signs/installs the module with no manual steps; the new kernel then boots with it already loaded, passing pattern tests and the full benchmark — see `RESEARCH.md` |
| Secure Boot with a self-signed module | ❓ | Test host currently has Secure Boot off |
| `kunlun_peermem.ko` (Mellanox GPUDirect RDMA) | 🚫 | Needs MLNX_OFED's IB peer-memory symbols, not present on the test host — correctly inert, not a bug |

### ML frameworks / serving stacks

| Capability | Status | Notes |
|---|:---:|---|
| Official `paddlepaddle-xpu` wheel — on-device compute (add/matmul/RNG) | ✅ | Numerically correct against real random data |
| Official `paddlepaddle-xpu` wheel — `paddle.to_tensor(host_array)` | ❌ | Silent no-op, **root cause found** via live `gdb` (no rebuild needed — `libpaddle.so` ships unstripped): the wheel's `Copy<phi::Place,phi::Place>` dispatcher correctly detects the CPU→XPU case but then calls the *wrong-direction* (D2H-shaped) copy overload instead of H2D, via Baidu-internal code that structurally diverges from the public Paddle source. Needs Baidu's own non-public source to actually fix — see `RESEARCH.md` |
| Fixing the wheel by pointing it at this repo's own `libxpuapi.so` | ❌ | Confirmed ABI-incompatible — fails at import (`undefined symbol`), both via `LD_LIBRARY_PATH` and file replacement |
| `vllm-kunlun` (Baidu's official vLLM plugin) | 🚫 | Kunlun3 (P800) only; K100/K200 unsupported, open feature request with no maintainer response |
| FastDeploy | 🚫 | Needs driver ≥5.0.21.26 on P800; this card's driver line (4.33.0) is unrelated and far older |
| PaddleOCR / PaddleOCR-VL / PaddleX | 🚫 | Target R200/R300/P800 only |
| Ollama / llama.cpp (`ggml`) | 🚫 | No Kunlun backend of any kind exists upstream or third-party |
| PaddleNLP llama2-7b tutorial, on the official wheel | ❌ | Dead end confirmed directly: wheel's `libxpuapi.so` does have `xpu1`-candidate RoPE/SwiGLU/RMSNorm/paged-attention symbols, but Paddle 2.6.1 has no XPU kernel registered for `fused_rotary_position_embedding` at all |

### Workload feasibility beyond LLM serving (static analysis of `libxpuapi.so`'s own symbol table, see `RESEARCH.md` "Other workloads")

| Capability | Status | Notes |
|---|:---:|---|
| CNN vision inference (ResNet-50-class, fixed-size attention) | ✅ | Already proven by this repo's own cases — [Key benchmarks](#key-benchmarks-k200-post-s9--cases) |
| Bare compute/RNG coprocessor (direct xdnn/XPURT calls, no framework) | ✅ | What this repo's own benchmarks/tests already do — lowest-risk path for anything new |
| K-means clustering | ⚠️ | Every needed primitive present and ungated (GEMM, `argmin`, one-hot trick for centroid update); untested end-to-end |
| CTR / ranking / recommendation (GRNN/DSSM/Wide&Deep-style) | ⚠️ | Looks like this card's original intended workload; untested on real weights |
| Classic object detection (YOLO-class via `yolo_unit_fusion`) | ⚠️ | Same envelope as proven CNN cases; untested |
| Classic (non-VL) OCR inference, pretrained (CRNN: conv+RNN+softmax) | ⚠️ | Inference looks feasible; **training** blocked (`ctc_loss_bwd` is gen-gated) |
| BERT/ERNIE-era encoder inference (hand-composed, unfused attention) | ⚠️ | GEMM+softmax+layer_norm all present; not confirmed end to end |
| Graph neural networks (message passing) | 🚫 | `index_add`/scatter-add is gen-gated (`xpu2::`/`xpu3::` only); no SpMM anywhere in this SDK regardless of generation |
| LLM transformer serving (RoPE/SwiGLU/RMSNorm/paged-attention) | 🚫 | Not implemented for xpu1 in this repo's `libxpuapi.so` at all (gen-gated to `xpu2::`/`xpu3::`); present-but-undispatched in the Paddle wheel's own build (see above) |
| Custom `ggml`/llama.cpp backend via hand-written cluster kernels | ⚠️ | A real hand-written xpu1 kernel (not SDK-provided) compiles cleanly with the real XTDK compiler and runs through launch with no errors on real hardware — but doesn't complete (`xpu_wait()` doesn't return; likely missing `Context` initialization a raw-XPURT launch skips, untested pending a separate toolchain fix). See `RESEARCH.md` and `docs/research/20261019-xpu1-custom-kernel-poc.md` |

**Maintenance:** this table should be updated in the same change as any work that confirms, breaks, or newly tests one of these capabilities — see `CLAUDE.md`'s per-change checklist.

---

## Key benchmarks (K200, post S9 + cases)

Measured on production knobs after S4–S9. Full write-ups under `docs/impl/` and `results/cases/`.

### Compute (xdnn GEMM)

| Precision | Peak | Efficiency | Notes |
|-----------|:----:|:----------:|-------|
| FP16 | ~20 TFLOPS | ~35% | `xdnn::fc`, 2048³, nc=4 |
| FP32 | ~7.8 TFLOPS | ~54% | `xdnn::fc`, 2048³ |
| INT8 | N/A | — | KL1 firmware has no INT8 CDNN kernel (S3) |

### PCIe bandwidth (XPURT `xpu_memcpy`)

| Direction | BW | Notes |
|-----------|:----:|-------|
| H2D pageable (S6/S9) | **~10.1 GB/s** | bounce pipeline (`kl1_bounce_pipe=1`) |
| D2H pageable (S6/S9) | **~7.8 GB/s** | S9 did not improve; use host_alloc for ~12.9 |
| H2D `xpu_host_alloc` (S4) | **~12.5 GB/s** | direct EDMA (`kl1_dma_direct=1`) |
| D2H `xpu_host_alloc` (S4) | **~12.9 GB/s** | ≤256MB; 1GB may fall back if hugepages exhausted |

### P2P (same card, PD0 → PD1)

| Metric | Value | Notes |
|--------|:-----:|-------|
| Bandwidth (S5) | **~11.2 GB/s** | was hang / ~2.5 GB/s |
| Data verify | PASS | `tests/test_p2p_verify` |
| Cross-card P2P | unsupported | host staging only if at all |

### Latency

| Operation | Latency |
|-----------|:------:|
| Kernel launch | ~2 µs |
| 1-thread dispatch | ~26 µs |
| 4-thread dispatch | ~25 µs |

### Inference / real cases

| Workload | Framework | Precision | Metric | Notes |
|----------|-----------|:---------:|--------|-------|
| **C1** ResNet-50 | Paddle 2.6.1 XPU | FP32 | b1 **~167** / b8–32 **~250** img/s | **current stack baseline** |
| ResNet-50 (historical) | older container/SDK | FP32 b1 | ~1069 img/s | not current; not a driver regression |
| **C2** Denoise 256² | native xdnn | FP16 | **~100 img/s** | pinned ≈ pageable (compute-bound) |
| **C3** Dual-PD + P2P | native | FP16 | dual **1.44 ms** vs solo **2.00 ms** (~1.39×) | same-card only |
| **C4** MLP tower e2e | native | FP16 host_alloc | b32 **~22.8k** img/s | ~1.8× vs FP32 pageable |
| **C5** Dual-card | 2× process | FP16 host_alloc | sum/single **~2.00×** | no cross-card P2P |

App defaults: `xpu_host_alloc` staging, **FP16** + `ncluster=4`, GEMM K ∈ [1024, 4096]. See [`docs/impl/20260715-s8-app-guidance.md`](docs/impl/20260715-s8-app-guidance.md).

---

## Driver fixes (KL1 / K200)

| Feature | Before | After (S4–S9) |
|---------|--------|---------------|
| `xpu_memcpy_peer` (same card) | hang / ~2.5 GB/s | **~11.2 GB/s**, verify PASS (S5) |
| `xpu_host_alloc` / `host_free` | -706 / -707 | PASS (2MB hugepage mmap, S2) |
| H2D/D2H via host_alloc | bounce only | **~12.5 / 12.9 GB/s** (S4) |
| pageable H2D/D2H | ~5.5 / ~4.9 | **~10.1 / ~7.8** (S6; S9 no D2H win) |

### Install

Two install scripts, same safety design (write `.ko`/register to disk first, then unload/reload; if unload sticks on *Unloading*, **reboot once** — the patched image is already on disk) but different durability:

| | `install_driver.sh` (manual) | `install_driver_dkms.sh` (real DKMS) |
|---|---|---|
| Installs from | a locally built `kunlun-driver/kunlun.ko` (run `make driver` first) | the committed `kunlun-driver/` tree at `HEAD` (via `git archive`; commit driver changes first) |
| Survives a reboot | yes | yes |
| Survives a **kernel upgrade** | **no** — rerun `make driver && make driver-install` by hand after every kernel bump | **yes** — registered with real DKMS (`dkms.conf`'s `AUTOINSTALL=yes`), so `apt`'s kernel-postinst hook auto-rebuilds and reinstalls it |
| Needs | nothing extra | the `dkms` package (`AUTO_INSTALL_DKMS=1` to let the script install it itself) |

```bash
# Manual path (no kernel-upgrade auto-rebuild):
make driver    # required after kernel package upgrade (vermagic must match uname -r)
sudo KL1_P2P_STUB=0 KL1_DMA_DIRECT=1 KL1_BOUNCE_PIPE=1 scripts/install_driver.sh
# or: make driver-install
# disk only (then reboot): sudo DISK_ONLY=1 scripts/install_driver.sh && sudo reboot
# install existing .ko without rebuild: make driver-install-norebuild
# install arbitrary path: make install-ko KO=/path/to/kunlun.ko

# Real-DKMS path (survives a kernel upgrade):
sudo KL1_P2P_STUB=0 KL1_DMA_DIRECT=1 KL1_BOUNCE_PIPE=1 AUTO_INSTALL_DKMS=1 scripts/install_driver_dkms.sh
# or: make driver-install-dkms AUTO_INSTALL_DKMS=1
# check registration any time: sudo dkms status

modinfo -F srcversion,vermagic kunlun
make regression
```

Both scripts take the same `KL1_P2P_STUB`/`KL1_DMA_DIRECT`/`KL1_BOUNCE_PIPE`/`DISK_ONLY`/`SKIP_MODPROBE_CONF` env vars and write the same `/etc/modprobe.d/kunlun-kl1.conf`. `install_driver_dkms.sh` additionally compresses the installed module on distros that do so by default (e.g. Ubuntu → `kunlun.ko.zst`) — this is handled transparently.

Boot-time options: `/etc/modprobe.d/kunlun-kl1.conf` (maintained by either script).  
Details: [`kunlun-driver/BUILD.md`](kunlun-driver/BUILD.md), closure §4.

### Runtime knobs (`/sys/module/kunlun/parameters/`)

| Param | Prod default | Notes |
|-------|:------------:|-------|
| `kl1_dma_direct` | 1 | S4 host_alloc direct EDMA |
| `kl1_bounce_pipe` | 1 | S6 pageable double-buffer |
| `kl1_pageable_pin` | **0** | keep off (4K pin hangs/slow on large xfers) |
| `kl1_p2p_stub` | 0 | debug path off |
| `kl1_bounce_d2h` | 1 | S9 default; no extra pageable D2H win |

---

## Kernel 7.0 / modern host support

The S0–S9 work above targeted kernel 6.8. This section covers building and running the same driver on kernel 6.15+ (checked on `7.0.0-38-generic`, gcc 15.2, Dell R730). Full background: [`docs/BUILD-linux-7.md`](docs/BUILD-linux-7.md) and [`RESEARCH.md`](RESEARCH.md), which also tracks everything still open.

### What works

| Layer | Result |
|-------|--------|
| Driver build | `kunlun.ko`, `kunlun_peermem.ko` build clean, `vermagic` matches `uname -r` |
| Driver load | Clean probe, `/dev/xpu0`, `/dev/xpu1`, `/dev/xpuctrl`, `/proc/xpu`, no `dmesg` warnings |
| DMA correctness | `tests/test_host_alloc`, `tests/test_pageable_verify` pattern checks both pass |
| GEMM compute (xdnn/XPURT direct) | FP16 ~19–22 TFLOPS at 33–39% eff. and FP32 7685 GFLOPS at 53.37% — matches the [Key benchmarks](#key-benchmarks-k200-post-s9--cases) above; INT8 correctly reports unsupported |
| PCIe bandwidth (explained) | Real, non-zero, delivering the expected efficiency **for the PCIe generation this specific test host's CPU actually supports**: ~half the repo's originally-documented peak (pageable H2D ~5.3 vs ~10.1 GB/s) because the link negotiates to Gen3 (8GT/s), not the card's rated Gen4 (16GT/s) — `lspci` confirms `LnkSta: 8GT/s (downgraded)` vs `LnkCap: 16GT/s`, root-caused to this host's CPU (Broadwell-EP) having no Gen4 PCIe controller at all. Not a driver bug; see `RESEARCH.md`. |

### What doesn't (yet)

| Layer | Result |
|-------|--------|
| Official `paddlepaddle-xpu` 2.6.1 wheel | Detects the card (`device_count()` → 2). **On-device compute (add, matmul, random generation) is actually correct.** The bug is narrower: `paddle.to_tensor(host_array)` — staging a numpy array onto the XPU — is a **silent no-op**; reads return stale content from a previous, unrelated device allocation, not zeros or corrupted data (confirmed by freeing a known-value tensor and immediately reading its memory back through an unrelated `to_tensor()` call). **Root cause found**, via live `gdb` against the installed wheel — no Paddle source rebuild needed, since `libpaddle.so` ships unstripped with full debug info. Traced the real call chain down to `paddle::memory::Copy<phi::Place, phi::Place>`, the runtime place-type dispatcher: confirmed via register inspection it correctly detects CPU→XPU (H2D), but that branch then calls the **wrong-direction overload** (`Copy<phi::CPUPlace, phi::XPUPlace>`, D2H-shaped, internally calling `MemcpySyncD2H`) instead of the needed H2D one — via code that routes through Paddle's generic CustomDevice plugin machinery rather than native XPU place construction, confirming this wheel's actual XPU-dispatch source is a Baidu-internal variant that structurally diverges from the public `PaddlePaddle/Paddle` tag, not a straight build of it. Driver, XPURT (tested directly via `ctypes`, bypassing Paddle — correctly round-trips H2D/D2H), and XDNN compute (add/matmul/RNG all correct) are all independently cleared. Not fixable by pointing the wheel at this repo's own `libxpuapi.so`: confirmed **ABI-incompatible** (fails at import, `undefined symbol` on a `layer_norm_grad` overload with a different signature between the two builds). The wheel's bundled library does have `xpu1`-candidate RoPE/SwiGLU/RMSNorm/paged-attention symbols this repo's copy lacks, but it's moot — Paddle 2.6.1 has no XPU kernel registered for `fused_rotary_position_embedding` at all (confirmed via a direct runtime call), so PaddleNLP's llama2-7b tutorial is a dead end on this wheel regardless. Actually fixing the dispatcher bug needs Baidu's own non-public source. See `RESEARCH.md` for the full isolation. |
| vLLM / FastDeploy / current PaddleOCR-XPU | **Not an option on this card at all**, independent of driver state: these all target Kunlun2 (R200/R300) or Kunlun3 (P800), a different runtime stack (`xpytorch`) with no KL1 equivalent. |

### Untested

- Loading under Secure Boot with a self-signed module (test host has Secure Boot off; the module is unsigned and taints the kernel on load).
- gcc 14 specifically (built and verified with gcc 15.2 plus the wrapper below). gcc 13 fails outright on a 7.0 kernel build (`-fmin-function-alignment` is rejected).

### Building

```bash
sudo apt install build-essential linux-headers-$(uname -r)

# gcc 15 defaults to C23, which breaks the conftest probes: force gnu11
printf '#!/bin/sh\nexec gcc -std=gnu11 "$@"\n' > ~/gcc11 && chmod +x ~/gcc11

cd kunlun-driver
make clean
rm -rf kunlun_module/conftest/*.h          # force the probes to run again
make modules CC=$HOME/gcc11 -j8            # -> kunlun.ko, kunlun_peermem.ko
modinfo -F vermagic ./kunlun.ko            # must match uname -r
cd ..
```

- Rebuild after every kernel upgrade; `vermagic` must match `uname -r`.
- To install, use `make driver-install-norebuild` (or `scripts/install_driver.sh`) on the `kunlun.ko` you just built — `make driver-install` depends on `make driver`, which rebuilds with the default `CC` and would fail the same way the stock tree does on 6.15+.
- **Load it by hand the first time** and read `dmesg` before installing it for boot: `sudo insmod kunlun-driver/kunlun.ko kl1_p2p_stub=0 kl1_dma_direct=1 kl1_bounce_pipe=1`.

What the kernel-7.0 patch does:

- **Kbuild:** kernel 6.15+ ignores `EXTRA_CFLAGS`, so it's forwarded to `ccflags-y`. Conftest probe flags gain `-std=gnu11 -fms-extensions` (without `-fms-extensions`, every probe that includes `pci.h` fails a `sizeof(struct filename)` assertion and the driver silently falls back to old-kernel code paths). `-Wno-error=int-conversion` restores the pre-gcc-14 warning behavior.
- **API renames:** `in_irq`→`in_hardirq`; `del_timer`→`timer_delete`; `hrtimer_init`+`.function`→`hrtimer_setup`; `nth_page(p, j)`→`p + j` (pages come from one contiguous allocation here); `follow_pfn`→`follow_pfnmap_start/end`.
- **Stubs for gcc-14-broken probes:** the "function is present" conftest probes are inverted and break once implicit declarations are hard errors, so `phys_to_dma` uses the plain address, `dma_is_direct` is assumed false, `dma_map_ops.map_resource` is treated as absent (peer-DMA check returns false), and the console-lock/`jiffies_to_timespec` branches are disabled; `vm_flags_set`/`vm_flags_clear` are always used (`vm_flags` const since 6.3). Confirmed correct by the GEMM/DMA-correctness results above.

Two unrelated repo gaps had to be fixed before any of this would build at all (both already fixed in this tree — see [Environment](#environment)): `libxpuapi.so` left as an unfetched Git LFS pointer, and the `Makefile` hardcoding an external SDK path and a specific developer's machine path.

---

## Project structure

```
.
├── benchmarks/                 # xpu_perf_test, app_pipeline, denoise, p2p pipeline, int8 probe
├── tests/                      # p2p / host_alloc / pageable verify
├── scripts/
│   ├── install_driver.sh       # manual install (disk-first, DISK_ONLY, modprobe.d) — no kernel-upgrade auto-rebuild
│   ├── install_driver_dkms.sh  # same, via real DKMS — survives a kernel upgrade (auto-rebuild)
│   ├── lib_driver_common.sh    # shared load/unload/verify functions for the two scripts above
│   ├── install_ko_file.sh
│   ├── run_driver_regression.sh
│   ├── run_real_cases.sh / run_demo.sh / run_c1…c5_*.sh
│   ├── check_phi_7220p.sh / phase1_mpss4_readiness.sh / phi_discover.py
│   └── try_build_mpss4_modules.sh
├── results/cases/              # C1–C5 reports + SUMMARY.md
├── docs/                       # research, plan, impl (closure, S4–S9, Phi plan)
├── kunlun-driver/              # Kernel module 4.33 + KL1 patches
├── xdnn-ubuntu_x86_64/         # xdnn SDK 2.0.0.725
├── third_party/                # local MPSS trees (gitignored)
├── README.md
└── README_cn.md
```

---

## Build & run

```bash
make all

./benchmarks/xpu_perf_test 0          # GEMM + bandwidth
./benchmarks/xpu_perf_test 0 bw       # bandwidth only

make benchmarks/xpu_app_pipeline && ./benchmarks/xpu_app_pipeline -d 0
# or: scripts/run_s8_app_bench.sh

make cases                            # C1–C5
make demo                             # quick C2/C3/C4
# DEMO_FULL=1 scripts/run_demo.sh
# DEMO_REGRESSION=1 scripts/run_demo.sh

make regression                       # S7 gate
make benchmarks/xpu_int8_probe && ./benchmarks/xpu_int8_probe 0
```

### Real cases

| Case | Command | Report |
|------|---------|--------|
| C1 ResNet-50 | `scripts/run_c1_resnet.sh` | `results/cases/c1_resnet50.md` |
| C2 Denoise | `scripts/run_c2_denoise.sh` | `results/cases/c2_denoise.md` |
| C3 Dual-PD P2P | `scripts/run_c3_p2p_pipeline.sh` | `results/cases/c3_p2p_pipeline.md` |
| C4 MLP tower | `scripts/run_c4_mlp.sh` | `results/cases/c4_mlp.md` |
| C5 Dual-card | `scripts/run_c5_dual_card.sh` | `results/cases/c5_dual_card.md` |

One-pager: [`docs/impl/20260716-demo-one-pager.md`](docs/impl/20260716-demo-one-pager.md) · Cases delivery: [`docs/impl/20260715-real-cases-c1-c2-c3.md`](docs/impl/20260715-real-cases-c1-c2-c3.md)

### Paddle inference

```bash
# Needs paddlepaddle-xpu (or podman path in run_c1)
python3 scripts/paddle_infer_benchmark.py --model paddle_models --batch 1,8,32
```

---

## Environment

| Component | Path / note |
|-----------|-------------|
| XPURT (bundled) | `xdnn-ubuntu_x86_64/so/libxpurt.so.1`, built against by default — no system install required (see [Kernel 7.0 / modern host support](#kernel-70--modern-host-support)) |
| XPURT (system install, optional) | `/usr/local/xpu-4.33.0/lib64/libxpurt.so.1`; override `XPURT_INC`/`XDNN_LIB` to use it instead |
| Driver (installed) | `/lib/modules/$(uname -r)/updates/dkms/kunlun.ko` — never shipped prebuilt; `*.ko` is gitignored, so `make driver` (or the 6.15+/7.0 build) is required on every host |
| xdnn | `xdnn-ubuntu_x86_64/` — `libxpuapi.so` is Git LFS; run `git lfs pull` after cloning |

---

## Out of scope

**K200 hardware / firmware**

- SM overclock (900 MHz locked)
- PCIe ×16 renegotiation beyond Gen4×8 on K200
- Hardware cross-card P2P / BAR IOVA
- INT8 GEMM without firmware update
- Further pageable D2H kernel iteration (S9 closed: no win)
- Default `kl1_pageable_pin=1`

**Phi**

- Treating 7220P as 7120P / MPSS 3
- Building MPSS4 modules against host kernel 6.8 as the supported path (expected fail; use Rocky 8 / 4.18)

---

## Historical baselines (pre-optimization)

May 2026 archives (P2P broken, ~4–5 GB/s DMA) — **not** current:

- [`results/PERF_SUMMARY.md`](results/PERF_SUMMARY.md) (2026-05-07)
- [`results/K200_ANALYSIS_REPORT.md`](results/K200_ANALYSIS_REPORT.md) (2026-05-08)
