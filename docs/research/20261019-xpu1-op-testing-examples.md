# K100/K200 real-hardware op testing: usage examples

Companion to `RESEARCH.md`'s "Hand-composed BERT/ERNIE-style attention", "`*_unit_fusion` convenience wrappers", and "`grnn_cell`" sections (all 2026-10-19). Each program below is self-contained, builds with plain `g++` (no XTDK needed — these call into the already-compiled `libxpuapi.so`, they don't write new device kernels), and was actually run against real K200 hardware to produce the results those sections describe.

Build pattern, same for all three:

```bash
g++ -std=c++17 -O2 \
  -I xdnn-ubuntu_x86_64/include -I kunlun-driver/include \
  -o <name> <name>.cpp \
  -L xdnn-ubuntu_x86_64/so -lxpuapi -lxpurt -Wl,-rpath,xdnn-ubuntu_x86_64/so
```

Run with `XPUAPI_DEBUG=1 ./<name>` to see the internal dispatch trace (an undocumented but real env var — found via `strings xdnn-ubuntu_x86_64/so/libxpuapi.so`).

## 1. Hand-composed BERT/ERNIE-style attention — working example

**Status: confirmed numerically correct.** This is the one to actually reuse. Single-head self-attention built entirely from individually-proven primitives — no fused `attention.h`/`page_attention.h` kernels (those are `xpu2::`/`xpu3::`-only).

```cpp
#include "xpu/xdnn.h"
#include <cstdio>
#include <cmath>
#include <vector>

namespace xdnn = baidu::xpu::api;

static float* dev_buf(const std::vector<float>& h) {
    float* d = nullptr;
    xpu_malloc((void**)&d, h.size() * sizeof(float));
    xpu_memcpy(d, h.data(), h.size() * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}
static float* dev_zeros(int n) {
    float* d = nullptr;
    xpu_malloc((void**)&d, n * sizeof(float));
    return d;
}

int main() {
    xpu_set_device(0);                      // required before create_context()
    auto* ctx = xdnn::create_context();

    const int64_t seq = 64, dim = 8;

    // NOTE: loop vars must be signed -- `(i*k)%5 - 2` underflows to a huge
    // unsigned value (then a huge float) if i is size_t and the modulo
    // result is 0 or 1. Use int64_t, not size_t, for any such pattern.
    std::vector<float> h_x(seq * dim), h_wq(dim * dim), h_wk(dim * dim),
        h_wv(dim * dim), h_wo(dim * dim), h_scale(dim, 1.0f), h_bias(dim, 0.0f);
    for (int64_t i = 0; i < (int64_t)h_x.size(); i++) h_x[i] = 0.01f * (float)(i % 7);
    for (int64_t i = 0; i < (int64_t)h_wq.size(); i++) h_wq[i] = 0.02f * (float)((i * 3) % 5 - 2);
    for (int64_t i = 0; i < (int64_t)h_wk.size(); i++) h_wk[i] = 0.02f * (float)((i * 5) % 5 - 2);
    for (int64_t i = 0; i < (int64_t)h_wv.size(); i++) h_wv[i] = 0.02f * (float)((i * 7) % 5 - 2);
    for (int64_t i = 0; i < (int64_t)h_wo.size(); i++) h_wo[i] = 0.02f * (float)((i * 11) % 5 - 2);

    float* d_x = dev_buf(h_x);
    float* d_wq = dev_buf(h_wq), *d_wk = dev_buf(h_wk), *d_wv = dev_buf(h_wv), *d_wo = dev_buf(h_wo);
    float* d_scale = dev_buf(h_scale), *d_bias = dev_buf(h_bias);
    float* d_q = dev_zeros(seq * dim), *d_k = dev_zeros(seq * dim), *d_v = dev_zeros(seq * dim);
    float* d_scores = dev_zeros(seq * seq), *d_probs = dev_zeros(seq * seq);
    float* d_ctx = dev_zeros(seq * dim), *d_proj = dev_zeros(seq * dim), *d_resid = dev_zeros(seq * dim);
    float* d_y = dev_zeros(seq * dim);
    float* d_mean = dev_zeros(seq), *d_var = dev_zeros(seq);

    // Q = X @ Wq, K = X @ Wk, V = X @ Wv  (plain fc -- nullptr max-ptrs are
    // fine here, same convention xpu_perf_test.cpp and xpu_denoise.cpp use)
    xdnn::fc<float, float, float, float>(ctx, d_x, d_wq, d_q, seq, dim, dim, false, false, nullptr, nullptr, nullptr);
    xdnn::fc<float, float, float, float>(ctx, d_x, d_wk, d_k, seq, dim, dim, false, false, nullptr, nullptr, nullptr);
    xdnn::fc<float, float, float, float>(ctx, d_x, d_wv, d_v, seq, dim, dim, false, false, nullptr, nullptr, nullptr);

    // scores = Q @ K^T  (w_trans=true avoids a manual transpose)
    xdnn::fc<float, float, float, float>(ctx, d_q, d_k, d_scores, seq, seq, dim, false, true, nullptr, nullptr, nullptr);

    // scale by 1/sqrt(dim), then softmax over the last axis
    xdnn::scale<float>(ctx, d_scores, d_scores, seq * seq, true, 1.0f / std::sqrt((float)dim), 0.0f);
    xdnn::softmax<float>(ctx, d_scores, d_probs, {seq, seq}, 1);

    // context = probs @ V
    xdnn::fc<float, float, float, float>(ctx, d_probs, d_v, d_ctx, seq, dim, seq, false, false, nullptr, nullptr, nullptr);

    // output projection + residual + layer norm
    xdnn::fc<float, float, float, float>(ctx, d_ctx, d_wo, d_proj, seq, dim, dim, false, false, nullptr, nullptr, nullptr);
    xdnn::add<float>(ctx, d_x, d_proj, d_resid, seq * dim);
    xdnn::layer_norm<float, float>(ctx, d_resid, d_y, seq, dim, 1e-5f, d_scale, d_bias, d_mean, d_var, false);

    xpu_wait();
    std::vector<float> h_y(seq * dim);
    xpu_memcpy(h_y.data(), d_y, h_y.size() * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    for (float v : h_y) printf("%.4f ", v);
    printf("\n");
    return 0;
}
```

Confirmed output is correctly normalized (~[-1.5, 1.5] range, matching LayerNorm's expected behavior) and every intermediate stage (Q/K/V magnitude, softmax row-sums = 1.0, etc.) checks out against hand-calculable expectations. See `RESEARCH.md` for the full debugging story (a signed/unsigned underflow bug in test code, not a hardware issue, was the only obstacle).

## 2. `yolo_unit_fusion` / `resnet_unit_fusion` — confirmed broken, for reference

**Status: both unimplemented for xpu1.** Included so anyone revisiting this doesn't have to rediscover the call shape from scratch. `yolo_unit_fusion` (inference mode, `is_train=false`):

```cpp
#include "xpu/xdnn.h"
#include <vector>
namespace xdnn = baidu::xpu::api;

static float* dev_const(int n, float v) {
    std::vector<float> h(n, v);
    float* d = nullptr;
    xpu_malloc((void**)&d, n * sizeof(float));
    xpu_memcpy(d, h.data(), n * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t N = 1, C = 3, H = 8, W = 8, F = 4;

    float* d_x = dev_const(N * C * H * W, 1.0f);
    float* d_w = dev_const(F * C * 3 * 3, 0.01f);
    float* d_y = nullptr; xpu_malloc((void**)&d_y, N * F * H * W * sizeof(float));
    float* d_scale = dev_const(F, 1.0f), *d_bias = dev_const(F, 0.0f);
    float* d_gmean = dev_const(F, 0.0f), *d_gvar = dev_const(F, 1.0f);  // inference-mode running stats
    float *d_bm = nullptr, *d_bis = nullptr;
    xpu_malloc((void**)&d_bm, F * sizeof(float));
    xpu_malloc((void**)&d_bis, F * sizeof(float));

    std::vector<std::vector<int64_t>> x_shape_list = {{N, C, H, W}};
    std::vector<xdnn::CVFusionConv2dParam> conv_param_list = {
        xdnn::CVFusionConv2dParam(F, C, /*group=*/1, {3, 3}, {1, 1}, {1, 1}, {1, 1})};
    std::vector<xdnn::CVFusionBatchNormParam> bn_param_list = {
        xdnn::CVFusionBatchNormParam(1e-5f, 0.9f, d_scale, d_bias, d_bm, d_bis, d_gmean, d_gvar)};
    std::vector<const float*> x_list = {d_x}, w_list = {d_w}, mx = {nullptr};

    // This is as far as it gets -- fails with XDNN_INVALID_PARAM, see below.
    int ret = xdnn::yolo_unit_fusion<float, float, float, float>(
        ctx, x_list, w_list, d_y, x_shape_list, conv_param_list, bn_param_list,
        mx, mx, nullptr, xdnn::Activation_t::LINEAR, /*is_nchw=*/true, /*is_train=*/false);
    printf("yolo_unit_fusion ret=%d\n", ret);  // -> 1 (XDNN_INVALID_PARAM)
    return ret;
}
```

With `XPUAPI_DEBUG=1`, this prints `gtest_yolo_unit_fusion<...>(api::kXPU1, ...)` then `[ASSERT-FAIL](0==1)[src/wrapper/fusion_yolo_unit_fusion.cpp:263]` — an unconditional "not implemented" gate, not a parameter mistake (confirmed by trying both `is_train` values with no change). `resnet_unit_fusion` hits the identical pattern at a different source line.

## 3. `grnn_cell` — correct calling convention, then a real hardware exception

**Status: dispatches for real, then faults the device. Do not run this without being prepared to reboot the host.** No public example ships with the SDK for this op; the calling convention below was reverse-engineered through three rounds of `XPUAPI_DEBUG=1` failures (see `RESEARCH.md` for the exact assert messages at each step).

```cpp
#include "xpu/xdnn.h"
#include <cmath>
#include <vector>
namespace xdnn = baidu::xpu::api;

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();

    const int64_t xdim = 3, hdim = 4;
    const int total_tokens = 5;             // two sequences, lengths 2 and 3
    std::vector<int> lod = {0, 2, 5};

    float* d_x = nullptr; xpu_malloc((void**)&d_x, total_tokens * xdim * sizeof(float));
    // ... xpu_memcpy real input data into d_x ...

    float* d_h_prev = nullptr;              // MUST be nullptr -- a real buffer here
                                             // trips an immediate assert.

    // GRNN needs 3 gate-weight matrices (update/reset/candidate, GRU-like),
    // and weight_x/weight_h must EACH be one contiguous allocation -- three
    // separately xpu_malloc'd buffers trip a pointer-equality assert
    // (it checks weight_h[0] + hdim*hdim*sizeof(short) == weight_h[1]).
    short *d_wx_base = nullptr, *d_wh_base = nullptr;
    xpu_malloc((void**)&d_wx_base, 3 * xdim * hdim * sizeof(short));
    xpu_malloc((void**)&d_wh_base, 3 * hdim * hdim * sizeof(short));
    // ... fill with quantized int16 weights (real = int16_val/32767*max_abs) ...
    std::vector<const short*> weight_x = {d_wx_base, d_wx_base + xdim * hdim, d_wx_base + 2 * xdim * hdim};
    std::vector<const short*> weight_h = {d_wh_base, d_wh_base + hdim * hdim, d_wh_base + 2 * hdim * hdim};

    float* d_wmax = nullptr; xpu_malloc((void**)&d_wmax, sizeof(float));
    // ... xpu_memcpy a real max-abs scale value into d_wmax ...
    std::vector<const float*> weight_x_max = {d_wmax, d_wmax, d_wmax};
    std::vector<const float*> weight_h_max = {d_wmax, d_wmax, d_wmax};

    float* d_y = nullptr; xpu_malloc((void**)&d_y, total_tokens * hdim * sizeof(float));
    xdnn::VectorParam<int> lod_param{lod.data(), (int64_t)lod.size(), nullptr};

    // With all of the above correct, this dispatches real internal
    // decomposition on api::kXPU1 (sequence_to_batch -> findmax ->
    // fc_fusion x3 -> batch_to_sequence) -- but then triggers a genuine
    // hardware exception (dmesg: "sse exception"/"task fifo overflow").
    // grnn_cell itself may even return 0 (success); the fault shows up
    // on the FOLLOWING xpu_wait() or allocation call.
    int ret = xdnn::grnn_cell<float, short, int, short>(
        ctx, d_x, d_h_prev, weight_x, weight_h, d_y, xdim, hdim, lod_param,
        nullptr, nullptr, weight_x_max, weight_h_max, nullptr);
    printf("grnn_cell ret=%d\n", ret);
    ret = xpu_wait();
    printf("xpu_wait ret=%d\n", ret);  // may return -713 or similar after a real fault
    return ret;
}
```

**If you run this and it faults**: `dmesg | tail` will show `kunlun: [WARN] xpu0: sse exception`/`xpu0 xtqN: task fifo overflow`. A module hot-reload (`scripts/install_driver.sh`) is **not sufficient** to recover — expect the next GEMM benchmark run to hang for several minutes and then report `OOM`. A full `sudo reboot` is required to restore the device to its normal, verified-healthy state.
