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

## Honest bottom line

This proves the XTDK-compiler-and-xpu1-target path is real and usable for writing genuinely new kernels (not just confirmed as "downloadable" — confirmed as "produces a correct device object the hardware accepts for loading"), which is meaningfully further than the compiler-availability-only conclusion reached earlier. It does **not** yet prove a hand-written kernel can be made to actually execute and return a correct result on K100/K200 — that remains open, with a concrete, scoped next step (get `baidu::xpu::api::Context`-based launch working, which first needs the `__float128`/`<limits>` issue solved, to test whether proper Context initialization is what's missing) rather than a vague "try harder."
