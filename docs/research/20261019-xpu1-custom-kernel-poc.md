# Hand-written xpu1 (K100/K200) cluster kernel: proof-of-concept

Status as of 2026-10-19: **compiles, links, and runs through kernel launch on real K200 hardware. Launch does not complete/signal within the tested wait window — root cause not yet isolated.** This is real, substantive progress toward the "custom `ggml`/llama.cpp backend via hand-written cluster kernels" idea (see `RESEARCH.md`), not a finished, working example — read the whole "Status" section before using this as a template.

## What this answers

TODO item: "have a look at custom ggml/llama.cpp backend via hand-written cluster kernels." The underlying open questions were:

1. Is Baidu's XTDK device compiler usable to build a *custom* (not xdnn-provided) kernel targeting xpu1 specifically? **Yes — confirmed.**
2. Does this repo's bundled plugin SDK (`xdnn-ubuntu_x86_64/plugin/`) actually support xpu1 as a target, or only xpu2/3? **xpu1 is a first-class target in the build system** (`search_and_compile_kernel("xpu1")`, `src/kernel/cpp/*.xpu`, `--xpu-arch=xpu1`), just with no example kernel shipped for it (only `kunlun2cpp/` examples exist).
3. Can a hand-written xpu1 kernel actually run correctly on real K200 hardware? **Not yet confirmed — got as far as a clean launch with no errors, but `xpu_wait()` doesn't return within the tested window.**

## The kernel

Flat/non-namespaced (xpu1's own convention — unlike the shipped `kunlun2cpp/add1.xpu` example, which wraps its kernel in `namespace xpu2 { }`). Deliberately scalar-only and single-threaded for this first correctness check:

```cpp
// add_two.xpu -- y[i] = x[i] + 2.0f, hand-written xpu1 cluster kernel.
//
// No __simd__ / vector SIMD types: xpu1 does not expose the 512-bit SIMD
// vector ALU that xpu2+ do. Confirmed two ways:
//   - cluster_simd.h (this repo's xdnn-ubuntu_x86_64/include/xpu/kernel/)
//     is guarded `#if (defined(__XPU3__) || defined(__XPU4__))` --- xpu1
//     and xpu2 are both excluded from that file entirely.
//   - Trying __simd__ local-memory buffers and svadd_float32x16() (used by
//     the shipped xpu2 example) against --xpu-arch=xpu1 fails to compile:
//     "unknown type name '__simd__'" / "undeclared identifier
//     'svadd_float32x16'", reported as "errors generated when compiling
//     for generic" -- xpu1 compiles to a lower-level "generic" profile,
//     not the same one xpu2 gets.
// GM2LM/LM2GM (global<->local memory DMA), core_id()/core_num()/
// cluster_id()/cluster_num(), and mfence() all compile and link fine for
// xpu1 -- the basic cluster-primitive API is intact, only the newer SIMD
// vector extensions are xpu2+-only.
#include "xpu/kernel/xtdk.h"

__global__ void add_two(const float* x, float* y, int len) {
    const int buf_size = 128;
    float local_x[buf_size];
    float local_y[buf_size];

    for (int i = 0; i < len; i += buf_size) {
        int read_len = (buf_size < len - i) ? buf_size : (len - i);
        GM2LM(x + i, local_x, read_len * sizeof(float));
        for (int k = 0; k < read_len; k++) {
            local_y[k] = local_x[k] + 2.0f;
        }
        mfence();
        LM2GM(local_y, y + i, read_len * sizeof(float));
    }
}
```

Build it for xpu1 directly (no CMake needed for just the kernel object):

```bash
XTDK=/path/to/xtdk-llvm15-<platform>_x86_64   # see RESEARCH.md "XTDK device compiler"
$XTDK/bin/clang++ -std=c++11 -O2 -c add_two.xpu \
  -D __XPU1__ --target=x86_64-linux-gnu --basename add_two \
  -fno-builtin --xpu-arch=xpu1 -fPIC \
  -Wno-int-to-void-pointer-cast -Wno-int-to-pointer-cast \
  -mllvm --xpu-inline-cost -mllvm --xpu-inline-hot-call \
  -I/path/to/xdnn-ubuntu_x86_64/include/
# produces: add_two.o  add_two.device.bin.o
```

This step is fully confirmed working — the device object compiles cleanly every time, with no warnings beyond the usual CMake dev-policy noise when driven through the full plugin `CMakeLists.txt`.

## Usage example (launcher)

Deliberately bypasses the full `baidu::xpu::api::Context`/XDNN C++ API and uses only the low-level XPURT API (`xpu/runtime.h`) — see "Toolchain issue" below for why, and note that this is also a reasonable design choice for a real custom backend (a from-scratch `ggml`/llama.cpp integration would most likely talk to XPURT directly anyway, not through Baidu's own C++ wrapper classes):

```cpp
// run_add_two.cpp
#include <xpu/runtime.h>
#include <cstdio>

__attribute__((global)) void add_two(const float* x, float* y, int len);

int main() {
    const int N = 1024;
    float host_in[N], host_out[N];
    for (int i = 0; i < N; i++) host_in[i] = (float)i;

    float* dev_x = nullptr;
    float* dev_y = nullptr;

    xpu_set_device(0);
    xpu_malloc((void**)&dev_x, N * sizeof(float));
    xpu_malloc((void**)&dev_y, N * sizeof(float));
    xpu_memcpy(dev_x, host_in, N * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    XPUStream stream = nullptr;
    xpu_stream_create(&stream);
    add_two<<<1, 1, stream>>>(dev_x, dev_y, N);
    int ret = xpu_wait(stream);   // <-- does not return within the tested window; see below
    if (ret != 0) { printf("xpu_wait failed: %d\n", ret); return 1; }

    xpu_memcpy(host_out, dev_y, N * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    for (int i = 0; i < N; i++) {
        if (host_out[i] != host_in[i] + 2.0f) { printf("MISMATCH at %d\n", i); return 1; }
    }
    printf("PASS\n");
    xpu_free(dev_x);
    xpu_free(dev_y);
    return 0;
}
```

Compile and link (same `XTDK`/SDK paths as above; `-x xpu` is required because of the `<<<...>>>` launch syntax, which only XTDK's clang understands):

```bash
$XTDK/bin/clang++ -std=c++11 -x xpu -c run_add_two.cpp \
  -D __XPU1__ -D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1 \
  --target=x86_64-linux-gnu --xpu-arch=xpu1 --basename run_add_two \
  -fno-builtin -fPIC -o run_add_two.o \
  -I/path/to/kunlun-driver/include
g++ -o run_add_two run_add_two.o add_two.o add_two.device.bin.o \
  -L/path/to/xdnn-ubuntu_x86_64/so -lxpurt \
  -Wl,-rpath,/path/to/xdnn-ubuntu_x86_64/so
./run_add_two
```

Running it against the real K200:
```
XPURT .../libxpurt.so.1 loaded
xpu_set_device -> 0
xpu_malloc(x) -> 0 ptr=0x208000000
xpu_malloc(y) -> 0 ptr=0x208020000
xpu_memcpy H2D -> 0
xpu_stream_create -> 0 stream=0x...
launching kernel...
launch issued, waiting...
[hangs here -- xpu_wait never returns in the tested window]
```

## Status: what's confirmed, what isn't

**Confirmed working:**
- Writing a real, non-trivial (not an SDK-provided) xpu1 kernel and compiling it with the real, publicly-downloaded XTDK compiler (`klx-sdk-release-public.su.bcebos.com`, see `RESEARCH.md`).
- `xpu_set_device`, `xpu_malloc` (×2), `xpu_memcpy` H2D, `xpu_stream_create` all succeed cleanly against real hardware, with real device addresses (`0x208...`, matching the pattern every other XPURT test in this repo's work has seen).
- The kernel launch call itself (`add_two<<<1,1,stream>>>(...)`) issues without an immediate error, and the device-side symbol name is correctly resolved (`_Z7add_twoPKfPfi`, confirmed via an earlier run's warning message before this one — demangles to `add_two(float const*, float*, int)`).

**Not yet confirmed:** the kernel actually executing to completion. `xpu_wait()` does not return within a 12–15s window, repeatably, across multiple kernel-source variants (including an extremely simple, deliberately single-threaded, no-multi-core-partitioning version, specifically to rule out a `core_num()`/`cluster_num()`-returns-zero self-inflicted infinite loop — same symptom either way). One earlier variant (with the original `<<<2,1>>>` multi-cluster launch, immediately after a prior test process had been force-killed mid-dispatch) returned a hardware error instead of hanging: `ioctl() fail, (713) Device abnormal due to previous error` — this was traced to the force-kill leaving the device's compute-dispatch state corrupted (confirmed via `dmesg`; a driver hot-reload cleared it, and `tests/test_host_alloc`/`test_pageable_verify` continued to pass throughout, so this never affected the DMA/memory path, only the compute-launch path tied to that specific process).

**Leading hypothesis, not yet tested:** every other working example of launching a kernel in this SDK (`xdnn-ubuntu_x86_64/plugin/src/wrapper/add2.cpp`, and this repo's own `benchmarks/`) goes through `baidu::xpu::api::Context` (created via `xdnn::create_context()`), not raw XPURT calls. `Context` construction may perform one-time compute-side initialization (e.g. HCM/cluster-partition setup) that this proof-of-concept's bare-XPURT approach skips entirely, which could be why a raw `xpu_malloc`/`xpu_memcpy`/`<<<>>>` sequence doesn't behave the same way xdnn's own GEMM kernels (which _do_ work, extensively proven elsewhere in this repo) do.

**Separate, unresolved toolchain issue found along the way (not fixed, blocks testing the hypothesis above directly):** compiling anything that pulls in the full `xpu/xdnn.h` (needed for `baidu::xpu::api::Context`) with this specific XTDK build (`2.7.85.2`, clang 15, 2021-era) against this host's very new glibc/libstdc++ (Ubuntu 26.04, glibc 2.43, gcc 15) hits two independent `__float128`-related incompatibilities:
  1. `bits/floatn.h`'s `__cfloat128` typedef uses `__attribute__((__mode__(__TC__)))`, which this XTDK clang rejects ("unsupported machine mode '__TC__'"). **Worked around**: glibc guards this specifically for `defined __clang__` cases with `(!defined __INTEL_LLVM_COMPILER || !defined SYCL_LANGUAGE_VERSION)`; passing `-D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1` flips that condition and skips the whole block. (Harmless for our purposes — we aren't using Intel's compiler or SYCL, just borrowing the macro check to dodge this specific code path.)
  2. libstdc++'s own `<limits>` separately specializes `numeric_limits<__float128>`, gated on `defined(_GLIBCXX_USE_FLOAT128)` — and this XTDK clang rejects `__float128` as a type outright ("`__float128` is not supported on this target"), a backend-level rejection, not just the TC-mode attribute syntax. **Not worked around**: tried `-D_GLIBCXX_USE_FLOAT128=0` (doesn't help — `#if defined(X)` is true regardless of X's value) and `-U__SIZEOF_FLOAT128__` (doesn't help either — `_GLIBCXX_USE_FLOAT128` is set by some other detection this doesn't affect). This is the same general *class* of problem as the gcc14/kernel-7.0 compatibility work already extensively documented in this repo (old vendor toolchain vs. bleeding-edge host) — this specific instance just hasn't been chased to a fix yet.

## Honest bottom line (as of 2026-10-19)

This proves the XTDK-compiler-and-xpu1-target path is real and usable for writing genuinely new kernels (not just confirmed as "downloadable" — confirmed as "produces a correct device object the hardware accepts for loading"), which is meaningfully further than the compiler-availability-only conclusion reached earlier. It does **not** yet prove a hand-written kernel can be made to actually execute and return a correct result on K100/K200 — that remains open, with a concrete, scoped next step (get `baidu::xpu::api::Context`-based launch working, which first needs the `__float128`/`<limits>` issue solved, to test whether proper Context initialization is what's missing) rather than a vague "try harder."

## Update 2026-10-20: the toolchain blocker is solved, the Context hypothesis is refuted, a better hypothesis found, and a more serious risk discovered

Picked this back up to actually test the leading hypothesis above. XTDK (`2.7.85.2`, `ubuntu2004` variant) was re-downloaded and extracted to `~/kunlun_build/xtdk/xtdk-llvm15-ubuntu2004_x86_64` on the R730 test host (kept there, not deleted, since it's expensive to re-fetch — ~2.4 GB extracted).

### The `__float128`/`<limits>` blocker: solved

Isolated with a minimal reproduction (`-x xpu --xpu-arch=xpu1`, progressively including `<cstdio>` then `<limits>` alone) rather than guessing from the full `xdnn.h` error output:

- **`<cstdio>` alone fails too**, not just `<limits>` — the problem is pervasive across the C++ standard library, not specific to `xdnn.h`. With the already-known `-D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1` workaround (for the `bits/floatn.h` `__cfloat128`/TC-mode issue), `<cstdio>` compiles clean but `<limits>` still fails.
- **Read the actual guard in this host's real header** (`/usr/include/x86_64-linux-gnu/c++/15/bits/c++config.h`) instead of guessing: `_GLIBCXX_USE_FLOAT128` is defined under `#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)`. The earlier attempt to fix this (`-D_GLIBCXX_USE_FLOAT128=0`, noted as not working in the original write-up above) couldn't have worked — the check is `defined()`, not value-based, and the real trigger is a different pair of macros entirely.
- **Confirmed via `clang++ -dM -E` that XTDK's clang predefines *both* `__FLOAT128__` and `__SIZEOF_FLOAT128__`** for this target, even under `-x xpu --xpu-arch=xpu1` — the frontend's target-info still reports standard x86_64 float128 availability, while the actual xpu1 backend can't codegen it. This is an internal frontend/backend inconsistency in Baidu's clang fork for the xpu target, not something glibc or libstdc++ did wrong.
- **Fix: `-U__FLOAT128__ -U__SIZEOF_FLOAT128__`** (undefining *both* — the earlier attempt only tried `-U__SIZEOF_FLOAT128__` alone, which can't work since it's an OR condition). With this plus the existing `-D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1`, `<limits>` compiles clean, and **the full `xpu/xdnn.h`** (pulling in `baidu::xpu::api::Context`, `create_context()`, `destroy_context()`) **compiles and links with no errors** against this host's Ubuntu 26.04/glibc 2.43/gcc 15 toolchain, under XTDK's 2021-era clang 15.

Full corrected compile line (adds two flags to the "Build it" examples above):
```bash
$XTDK/bin/clang++ -std=c++11 -x xpu -c your_file.cpp \
  -D __XPU1__ -D__INTEL_LLVM_COMPILER=1 -DSYCL_LANGUAGE_VERSION=1 \
  -U__FLOAT128__ -U__SIZEOF_FLOAT128__ \
  --target=x86_64-linux-gnu --xpu-arch=xpu1 --basename your_file \
  -fno-builtin -fPIC -o your_file.o \
  -I /path/to/xdnn-ubuntu_x86_64/include -I /path/to/kunlun-driver/include
```

### The Context hypothesis: tested directly, refuted

Built a second launcher, identical to the original `run_add_two.cpp` except using `baidu::xpu::api::create_context()` and launching on `ctx->get_stream()` instead of a bare `xpu_stream_create()` stream. **Result: the exact same hang.** `xpu_wait()` does not return within the tested window (20–30s), regardless of whether the stream came from a `Context` or was created raw.

One detail worth recording: `ctx->get_stream()` on a freshly-created `Context` returned `nullptr` — a default `Context` doesn't have a stream until `set_stream()` is called. The launch was still accepted with a null stream in both the `<<<1,1,...>>>` and `<<<1,64,...>>>` configurations below; it just never completed either way.

**This refutes the original leading hypothesis.** `Context` construction is not the missing piece — whatever's wrong is something else.

### A better-supported hypothesis: missing work partitioning

Compared against the SDK's own one shipped, actually-used-by-its-test-suite reference kernel (`xdnn-ubuntu_x86_64/plugin/src/kernel/kunlun2cpp/add1.xpu`, launched from `src/wrapper/add2.cpp` as `xpu2::add1<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(...)`). Two differences from this proof-of-concept's kernel:

1. **Launch config**: the reference uses `ctx->ncluster()` clusters × **64** threads; this PoC's kernel was launched `<<<1, 1, ...>>>` — a single thread.
2. **Work partitioning, the more likely culprit**: the reference kernel computes `core_id()`/`core_num()`/`cluster_id()`/`cluster_num()`, derives a `thread_id`/`nthreads`, and **strides its loop accordingly**, with an early-return guard (`if (cid >= ncores) return;`). This proof-of-concept's `add_two` kernel has **no such partitioning at all** — every launched thread runs the identical, full-range loop over the whole buffer, redundantly.

Retried with `<<<1, 64, stream>>>` (bumping thread count to match the reference's block size) **without** adding partitioning logic to the kernel body — **same hang.** This doesn't disprove the partitioning hypothesis (64 unpartitioned threads all hammering the same `GM2LM`/`LM2GM` DMA ranges concurrently is still a plausible hang cause; more threads without fixing the actual redundant-work structure wouldn't necessarily help and could even make contention worse). It does rule out "just launch more threads" as a one-line fix. **Not yet tried**: actually adding `core_id()`/`cluster_id()`-based striding to the kernel body, matching the reference's structure. This is the next concrete step.

### A more serious risk, discovered along the way

The first `<<<1,1,...>>>` timeout (30s, then force-killed by the test harness's `timeout`) left the device in a state a plain driver reload (`rmmod`+`insmod`) **did** clear — matching this doc's original note above. But the *second* attempt (`<<<1,64,...>>>`, 20s timeout, force-killed the same way) left the device's compute PD in a **persistent `state=ERROR`** (confirmed via `/proc/xpu/dev0`, which also showed a stuck, unrelated `findmax_sdnn` task from a later GEMM-benchmark health-check — i.e. the error state blocks *all* subsequent compute, not just a retry of the same kernel). This time, `rmmod`+`insmod` **did not clear it** — the next full `xpu_perf_test` GEMM run hung for the full 200s timeout. **A full `sudo reboot` was required**, after which the device returned to its normal, GEMM-baseline-verified-healthy state.

**This means hand-written custom-kernel testing on this card is now confirmed to carry the same severity of risk as `grnn_cell`** (see `RESEARCH.md`'s `grnn_cell` section) — not a freely-retestable activity. Concretely:
- After any custom-kernel test that times out, **do not assume a module reload is sufficient recovery** — verify with a real compute operation (the full `xpu_perf_test` GEMM suite, not just `probe_error` or a DMA-only test like `test_host_alloc`, which can stay healthy while compute is genuinely stuck).
- If GEMM is also stuck after a reload, a full reboot is the fix, the same as `grnn_cell` required.
- Treat every further custom-kernel iteration as carrying this cost, and get explicit sign-off before attempting one, rather than iterating casually.

## Update 2026-10-20 (continued): the partitioning hypothesis tested too — also refuted, plus a correction

Tested the next concrete step directly: a second kernel (`add_two_v2.xpu`), identical to the first except adding real `core_id()`/`core_num()`/`cluster_id()`/`cluster_num()`-based work partitioning (each thread handles a distinct strided slice, with an early-return guard), launched exactly as the reference does — `<<<ctx->ncluster(), 64, stream>>>` (confirmed `ctx->ncluster()` is `4` for this device, printed at runtime, not assumed).

**Result: identical hang.** Same symptom as both earlier attempts — `xpu_wait()` doesn't return within the timeout window. At the time, the device appeared to stay healthy afterward (confirmed with a real GEMM run) — **this turned out to be wrong, found later while testing `fft` for the audio/speech investigation.** The `add_two_v2` task was in fact left wedged in hardware task queue `tq6` (`/proc/xpu/dev0/tqinfo` showed `tq6: state=HANG`, the original task still at its head, ten tasks deep by the time it was found) — the GEMM check that seemed to confirm health had simply been scheduled onto a different, still-idle queue and never touched the wedged one. **Running the full GEMM suite does not prove the device is clean; it only proves some queue works.** The real check is `cat /proc/xpu/dev0/tqinfo` and confirming all 8 queues report `NORMAL`. A plain driver reload did clear it once this was found (verified via `tqinfo`, all 8 `NORMAL`) — so a full reboot wasn't needed in this instance, but that's now a less certain claim than it looked at the time, since the method that "confirmed" it before was the same flawed one.

**This refutes the partitioning hypothesis too.** Three independent variables have now been tested without changing the outcome: bare XPURT vs. `Context`-based launch; a degenerate `<<<1,1,...>>>` launch vs. matching the reference's exact `<<<ncluster,64,...>>>` config; and an unpartitioned kernel body vs. one with real `core_id()`/`cluster_id()` striding matching the reference's structure. None of it mattered — every variant hangs the same way.

**Important correction, found while re-examining the "reference" kernel used for comparison:** `add1.xpu` (the one kernel this whole comparison was based on) is wrapped in `namespace xpu2 { ... }` — it targets **xpu2 hardware specifically, not xpu1**. This card is xpu1 (K100/K200). **There is no shipped, confirmed-working example anywhere in this SDK of a raw hand-written `<<<>>>` kernel launch actually completing on xpu1** — the only reference this proof-of-concept had was for a different hardware generation the whole time. That changes the picture: it's no longer "a known-good pattern that this proof-of-concept deviates from," it's "no known-good pattern for xpu1 exists in this SDK at all."

**Updated leading hypothesis:** given the kernel genuinely reaches the hardware (bound to a real task queue, `xtq6`, with correctly-marshalled parameters — confirmed in the first round of testing above) but never signals completion, regardless of launch config or kernel structure, the most likely explanation is now that **raw custom-kernel dispatch and completion-signaling for xpu1 was simply never implemented or validated by Baidu** (xpu2+ got a shipped example and presumably internal test coverage; xpu1 did not) — a firmware/runtime gap specific to this generation, not a mistake in how this proof-of-concept wrote or launched its kernel. This is a hypothesis, not a proven fact — it would take either a lower-level trace (comparing the exact hardware dispatch descriptor xdnn's own working kernels use vs. what a raw `<<<>>>` launch produces) or confirmation from Baidu to establish for certain, and neither has been attempted.

**On a possible power-related explanation (asked and checked 2026-10-20):** the test host's PCIe slot reports `SlotPowerLimit 25W` (`lspci -vvv`) — low for an accelerator card, and worth a physical check (Dell R730 PCIe risers are documented to have jumper-configurable power modes in some configurations). But the evidence from this testing argues against power being the cause of this specific hang: the heaviest, most power-hungry workload tested (the full GEMM benchmark, run dozens of times throughout this whole investigation) has never faulted, while the thing that does hang (a few-dozen-float custom kernel) is trivially light compute; zero PCIe AER/link-error events were ever logged; and `reset_count` stayed at 0. A genuine power-delivery problem would more plausibly show up under heavy sustained load than a near-idle one. Live power telemetry (the driver's `xpu_monitor.c`/`xpu_read_power` is internal-ioctl only, not exposed via the public XPURT API or `/proc/xpu`) wasn't pulled, so this isn't fully closed out quantitatively — but structurally, this looks like a software/firmware gap, not a power one.

### Updated honest bottom line (2026-10-20, final)

The compiler-and-toolchain path is fully clear, including for `Context`-based code. Three specific hypotheses for the `xpu_wait()` hang were tested directly on real hardware and all three were refuted: missing `Context` initialization, a degenerate launch configuration, and missing work partitioning. The weight of evidence now points toward raw custom-kernel dispatch on xpu1 being an unimplemented or unvalidated path in Baidu's own runtime/firmware for this specific generation, rather than anything fixable by writing the kernel differently — though this is the leading hypothesis, not a confirmed fact. Getting further would need either a lower-level dispatch-descriptor comparison against a known-working xdnn-internal kernel, or input from Baidu directly. Every attempt so far has landed somewhere between "clean hang, no lasting harm" and "device needs a full reboot" — never a clean success — so further iteration here should be treated as diminishing-returns exploration, not a likely-to-succeed next step, unless a genuinely new angle (the dispatch-descriptor comparison, or outside information) is available. **One correction found afterward**: the "clean hang, no lasting harm" end of that range was itself verified with a method later found insufficient — see the correction above. Anyone revisiting this should check `/proc/xpu/dev<N>/tqinfo` for all 8 queues reading `NORMAL`, not just run the GEMM suite once.
