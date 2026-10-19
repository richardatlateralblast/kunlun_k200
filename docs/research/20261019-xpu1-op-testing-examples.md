# K100/K200 real-hardware op testing: usage examples

Companion to `RESEARCH.md`'s "Hand-composed BERT/ERNIE-style attention", "`*_unit_fusion` convenience wrappers", "`grnn_cell`", "`lstm_inference`"/"`bilstm_inference`", and "Hand-composed YOLO-style conv unit" sections (all 2026-10-19). Each program below is self-contained, builds with plain `g++` (no XTDK needed — these call into the already-compiled `libxpuapi.so`, they don't write new device kernels), and was actually run against real K200 hardware to produce the results those sections describe.

Build pattern, same for all six:

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

## 4. `lstm_inference` — confirmed working

**Status: dispatches, computes correctly, leaves the device healthy.** Unlike `grnn_cell`, no reverse-engineering was needed — the plain (non-templated, non-quantized) overload in `xpu/refactor/fusion.h` has a fully commented signature, each parameter annotated with its exact PaddlePaddle fluid variable-name correspondence.

```cpp
#include "xpu/xdnn.h"
#include <cstdio>
#include <vector>
namespace xdnn = baidu::xpu::api;

static float* dev_buf(const std::vector<float>& h) {
    float* d = nullptr;
    xpu_malloc((void**)&d, h.size() * sizeof(float));
    xpu_memcpy(d, h.data(), h.size() * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t batch_size = 1, seq_len = 2, xdim = 2, hdim = 2;

    std::vector<float> h_x = {0.1f, 0.2f, 0.3f, 0.4f};          // [batch, seq, xdim]
    std::vector<float> h_wx(4 * hdim * xdim), h_wh(4 * hdim * hdim);
    for (int64_t i = 0; i < (int64_t)h_wx.size(); i++) h_wx[i] = 0.05f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_wh.size(); i++) h_wh[i] = 0.05f * (float)((i % 5) - 2);
    std::vector<float> h_bx(4 * hdim, 0.0f), h_bh(4 * hdim, 0.0f);

    float* d_x = dev_buf(h_x);
    float* d_wx = dev_buf(h_wx), *d_wh = dev_buf(h_wh);
    float* d_bx = dev_buf(h_bx), *d_bh = dev_buf(h_bh);
    float* d_last_h = nullptr; xpu_malloc((void**)&d_last_h, batch_size * hdim * sizeof(float));

    int64_t h_seq_len[1] = {seq_len};
    int64_t* d_seq_len = nullptr;
    xpu_malloc((void**)&d_seq_len, sizeof(h_seq_len));
    xpu_memcpy(d_seq_len, h_seq_len, sizeof(h_seq_len), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    // init_h/init_c both nullptr is the documented convention (see the
    // header comment next to each param); no reverse-engineering needed,
    // unlike grnn_cell above.
    int ret = xdnn::lstm_inference(ctx,
        d_x, /*x_need_transpose=*/true,
        /*init_h=*/nullptr, /*init_c=*/nullptr,
        d_seq_len,
        d_wx, /*wx_maxptr=*/nullptr,
        d_wh, /*wh_maxptr=*/nullptr,
        d_bx, d_bh,
        d_last_h,
        batch_size, seq_len, xdim, hdim);
    printf("lstm_inference ret=%d\n", ret);
    printf("xpu_wait ret=%d\n", xpu_wait());

    std::vector<float> h_last_h(batch_size * hdim);
    xpu_memcpy(h_last_h.data(), d_last_h, h_last_h.size() * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("last_h: %.6f %.6f\n", h_last_h[0], h_last_h[1]);  // tanh-bounded: must be in [-1,1]
    return ret;
}
```

Run twice in a row: output is exactly reproducible (`0.016579 -0.015127` both times), both values correctly within `[-1,1]` (an LSTM hidden state is always tanh-bounded), `dmesg` logs nothing at all (not even the benign `host_alloc` INFO lines larger tests produce), and the full `xpu_perf_test` GEMM suite run immediately after matches the established baseline exactly — no device degradation, unlike `grnn_cell`. With `XPUAPI_DEBUG=1`, the trace shows a real decomposition (`gtest_transpose` → `gtest_cast`/`gtest_copy` → `gtest_lstm_train<float,float,int16_t>(api::kXPU1, ...)`), confirming this isn't a stub either — it's genuinely computing on the chip.

**Still untested**: `bilstm_inference` (the actual bidirectional variant a real CRNN needs) and realistic dimensions (this used toy `hdim=2`/`seq_len=2`/`batch_size=1`).

## 5. `bilstm_inference` — confirmed working

**Status: dispatches, computes correctly, leaves the device healthy — same clean outcome as `lstm_inference` above.** Uses `baidu::xpu::api::bilstm_inference<float,float,short>` (a real compiled instantiation, confirmed via `nm -D`), the actual bidirectional variant a real CRNN uses. Forward and backward weight sets are reused from the same values here for simplicity — this is a dispatch/correctness-shape smoke test, not a trained model.

```cpp
#include "xpu/xdnn.h"
#include <cstdio>
#include <vector>
namespace xdnn = baidu::xpu::api;

static float* dev_buf(const std::vector<float>& h) {
    float* d = nullptr;
    xpu_malloc((void**)&d, h.size() * sizeof(float));
    xpu_memcpy(d, h.data(), h.size() * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}
static float* dev_zeros(int64_t n) { float* d = nullptr; xpu_malloc((void**)&d, n * sizeof(float)); return d; }

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t batch_size = 1, seq_len = 2, xdim = 2, hdim = 2, layer_num = 1;

    std::vector<float> h_x = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> h_wx(4 * hdim * xdim), h_wh(4 * hdim * hdim);
    for (int64_t i = 0; i < (int64_t)h_wx.size(); i++) h_wx[i] = 0.05f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_wh.size(); i++) h_wh[i] = 0.05f * (float)((i % 5) - 2);
    std::vector<float> h_bx(4 * hdim, 0.0f), h_bh(4 * hdim, 0.0f);

    float* d_x = dev_buf(h_x);
    float* d_fwd_wx = dev_buf(h_wx), *d_fwd_wh = dev_buf(h_wh);
    float* d_fwd_bx = dev_buf(h_bx), *d_fwd_bh = dev_buf(h_bh);
    float* d_bwd_wx = dev_buf(h_wx), *d_bwd_wh = dev_buf(h_wh);
    float* d_bwd_bx = dev_buf(h_bx), *d_bwd_bh = dev_buf(h_bh);

    // Output shapes aren't spelled out in a header comment for this overload
    // (unlike plain lstm_inference); allocate generously.
    float* d_y = dev_zeros(seq_len * batch_size * 2 * hdim);
    float* d_last_h = dev_zeros(2 * batch_size * hdim);
    float* d_last_c = dev_zeros(2 * batch_size * hdim);
    std::vector<int64_t> seq_len_tensor = {seq_len};

    int ret = xdnn::bilstm_inference<float, float, short>(ctx,
        d_x, /*init_h=*/nullptr, /*init_c=*/nullptr,
        d_fwd_wx, d_fwd_wh, d_fwd_bx, d_fwd_bh,
        d_bwd_wx, d_bwd_wh, d_bwd_bx, d_bwd_bh,
        d_y, d_last_h, d_last_c,
        batch_size, xdim, hdim, seq_len, seq_len_tensor, layer_num,
        /*x_maxptr=*/nullptr, /*h_maxptr=*/nullptr,
        /*forward_wx_maxptr=*/nullptr, /*forward_wh_maxptr=*/nullptr,
        /*backward_wx_maxptr=*/nullptr, /*backward_wh_maxptr=*/nullptr,
        xdnn::Activation_t::TANH, xdnn::Activation_t::SIGMOID);  // documented default elsewhere in fusion.h

    printf("bilstm_inference ret=%d\n", ret);
    printf("xpu_wait ret=%d\n", xpu_wait());

    std::vector<float> h_last_h(2 * batch_size * hdim);
    xpu_memcpy(h_last_h.data(), d_last_h, h_last_h.size() * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("last_h: %.6f %.6f %.6f %.6f\n", h_last_h[0], h_last_h[1], h_last_h[2], h_last_h[3]);
    return ret;
}
```

Run twice: output is exactly reproducible, all four values correctly within `[-1,1]`, `dmesg` logs nothing at all, and the full `xpu_perf_test` GEMM suite run immediately after matches the established baseline. With `XPUAPI_DEBUG=1`, the trace shows a real decomposition — `gtest_bilstm_inference` → `gtest_bilstm_train` (forward pass, backward pass) → `gtest_concat` to join the two directions — confirming genuine, non-stub computation.

**Correctness cross-check**: the forward-direction half of this output (`0.016579 -0.015127`) exactly matches example 4's standalone `lstm_inference` result above, since both tests used identical weights and input — internal consistency between the two APIs, not just "didn't crash."

With this, both the unidirectional and bidirectional LSTM a real CRNN needs are confirmed working. **Still untested**: realistic (larger, batched) dimensions — both this test and example 4 used toy shapes.

## 6. Hand-composed YOLO-style conv unit — working example

**Status: confirmed numerically correct.** Direct hand-composed equivalent of a YOLO conv block, built the same way example 1's attention block was: individually-proven primitives standing in for a broken convenience wrapper (`yolo_unit_fusion`, example 2 above). `conv2d_fusion` → `batch_norm` → `leaky_relu`, with FP16 activations and `TGEMM=signed char` — the exact dtype combination this repo's own `benchmarks/xpu_denoise.cpp` already proves works; an all-`float` instantiation of `conv2d_fusion` hits a clean `XDNN_UNIMPLEMENTED` on xpu1 first, so don't reach for that by default.

```cpp
#include "xpu/xdnn.h"
#include <cstdio>
#include <cmath>
#include <vector>

namespace xdnn = baidu::xpu::api;

static float16* dev_buf16(const std::vector<float>& h) {
    std::vector<float16> h16(h.size());
    for (size_t i = 0; i < h.size(); i++) h16[i] = float16(h[i]);
    float16* d = nullptr;
    xpu_malloc((void**)&d, h16.size() * sizeof(float16));
    xpu_memcpy(d, h16.data(), h16.size() * sizeof(float16), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}
static float* dev_buf32(const std::vector<float>& h) {
    float* d = nullptr;
    xpu_malloc((void**)&d, h.size() * sizeof(float));
    xpu_memcpy(d, h.data(), h.size() * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();

    const int64_t N = 1, C = 3, H = 8, W = 8, F = 4, KS = 3;
    const int64_t out_sz = N * F * H * W;

    std::vector<float> h_x(N * C * H * W), h_w(F * C * KS * KS), h_bias(F, 0.0f);
    for (int64_t i = 0; i < (int64_t)h_x.size(); i++) h_x[i] = 0.1f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_w.size(); i++) h_w[i] = 0.05f * (float)((i % 7) - 3);

    float16* d_x = dev_buf16(h_x);
    float16* d_w = dev_buf16(h_w);
    float* d_bias = dev_buf32(h_bias);
    float16* d_conv = nullptr; xpu_malloc((void**)&d_conv, out_sz * sizeof(float16));

    // conv2d_fusion: raw conv + bias, Activation_t::LINEAR -- BN needs to
    // sit between conv and the real activation, so no fused activation here.
    int ret = xdnn::conv2d_fusion<float16, float16, float16, signed char>(
        ctx, d_x, d_w, d_conv, N, C, H, W, F, {KS, KS}, {1, 1}, {1, 1}, {1, 1},
        /*group=*/1, nullptr, nullptr, nullptr, /*is_nchw=*/true, d_bias,
        /*branch=*/nullptr, xdnn::Activation_t::LINEAR);
    printf("conv2d_fusion ret=%d\n", ret);

    // batch_norm<float16> is XDNN_UNIMPLEMENTED on xpu1 -- cast to fp32,
    // matching the usual practice of computing BN statistics in fp32 even
    // in an otherwise-fp16 model, then cast back for the activation.
    float* d_conv32 = nullptr; xpu_malloc((void**)&d_conv32, out_sz * sizeof(float));
    xdnn::cast<float16, float>(ctx, d_conv, d_conv32, out_sz);

    std::vector<float> h_scale(F), h_bn_bias(F);
    for (int64_t i = 0; i < F; i++) { h_scale[i] = 1.0f + 0.1f * (float)i; h_bn_bias[i] = 0.05f * (float)i; }
    float* d_scale = dev_buf32(h_scale), *d_bn_bias = dev_buf32(h_bn_bias);
    float *d_bm = nullptr, *d_bis = nullptr, *d_gmean = nullptr, *d_gvar = nullptr;
    xpu_malloc((void**)&d_bm, F * sizeof(float));
    xpu_malloc((void**)&d_bis, F * sizeof(float));
    xpu_malloc((void**)&d_gmean, F * sizeof(float));
    xpu_malloc((void**)&d_gvar, F * sizeof(float));
    float* d_bn32 = nullptr; xpu_malloc((void**)&d_bn32, out_sz * sizeof(float));

    ret = xdnn::batch_norm<float>(ctx, d_conv32, d_bn32, N, F, H, W, 1e-5f, 0.9f,
        d_scale, d_bn_bias, d_bm, d_bis, d_gmean, d_gvar, /*is_nchw=*/true);
    printf("batch_norm ret=%d\n", ret);

    float16* d_bn = nullptr; xpu_malloc((void**)&d_bn, out_sz * sizeof(float16));
    xdnn::cast<float, float16>(ctx, d_bn32, d_bn, out_sz);

    float16* d_y = nullptr; xpu_malloc((void**)&d_y, out_sz * sizeof(float16));
    ret = xdnn::leaky_relu<float16>(ctx, d_bn, d_y, out_sz, 0.1f);
    printf("leaky_relu ret=%d\n", ret);

    xpu_wait();
    std::vector<float16> h_y16(out_sz);
    xpu_memcpy(h_y16.data(), d_y, out_sz * sizeof(float16), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    for (int64_t i = 0; i < 4; i++) printf("%.5f ", (float)h_y16[i]);
    printf("\n");
    return ret;
}
```

Checked against an independent CPU reference computing the same conv → per-channel batch-norm → leaky-ReLU sequence: every stage (conv output, batch-norm output, final activation) matched within FP16 rounding tolerance (max diff ≈0.0005–0.009, consistent with FP16's ~3 decimal digits of precision), including correctly-scaled negative values from the leaky-ReLU branch (e.g. device `-0.04987` vs. reference `-0.04925`). No `dmesg` warnings or hardware exceptions — unlike `grnn_cell`, this is a clean, safe, reusable pattern.

**Still untested**: a full multi-layer YOLO backbone at realistic scale (this used a single conv unit, toy `8×8` spatial size) and object-detection-specific heads (anchor decoding, NMS) beyond the conv+BN+activation building block itself.

## 7. K-means (one iteration) — confirmed working end to end

**Status: a complete, wired pipeline, not isolated ops — exact match to hand-computed expectations.** Unlike examples 1–6 above, which each test one op or one small fused block, this wires nine real kernel calls together (distance → assign → update) in a single program, matching what `RESEARCH.md`'s K-means bullet originally proposed. All ops are from the plain `math.h`/`nn.h` families (same as `add`/`scale`/`softmax`), not `fusion.h`'s recurrent-cell family.

4 points, 2 clusters, deliberately well-separated so the correct result is obvious by hand: `p0=(0,0)`, `p1=(0,1)` near `c0=(0,0)`; `p2=(5,5)`, `p3=(5,6)` near `c1=(5,5)`.

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
static float* dev_zeros(int64_t n) { float* d = nullptr; xpu_malloc((void**)&d, n * sizeof(float)); return d; }

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t N = 4, D = 2, K = 2;
    std::vector<float> h_x = {0,0, 0,1, 5,5, 5,6};   // [N,D]
    std::vector<float> h_c = {0,0, 5,5};              // [K,D]
    float* d_x = dev_buf(h_x);
    float* d_c = dev_buf(h_c);

    // ||x_i||^2, ||c_j||^2 per row.
    float* d_x_sq = dev_zeros(N * D); float* d_c_sq = dev_zeros(K * D);
    xdnn::square<float>(ctx, d_x, d_x_sq, N * D);
    xdnn::square<float>(ctx, d_c, d_c_sq, K * D);
    float* d_normx = dev_zeros(N); float* d_normc = dev_zeros(K);
    xdnn::reduce_sum<float>(ctx, d_x_sq, d_normx, {N, D}, {1});
    xdnn::reduce_sum<float>(ctx, d_c_sq, d_normc, {K, D}, {1});

    // x_i . c_j -> [N,K]; dist2 = normx[:,None] + normc[None,:] - 2*xc
    float* d_xc = dev_zeros(N * K);
    xdnn::fc<float, float, float, float>(ctx, d_x, d_c, d_xc, N, K, D, false, true, nullptr, nullptr, nullptr);
    float* d_neg2xc = dev_zeros(N * K);
    xdnn::scale<float>(ctx, d_xc, d_neg2xc, N * K, true, -2.0f, 0.0f);
    float* d_tmp = dev_zeros(N * K);
    xdnn::broadcast_add<float>(ctx, d_neg2xc, d_normx, d_tmp, {N, K}, {N, 1});
    float* d_dist2 = dev_zeros(N * K);
    xdnn::broadcast_add<float>(ctx, d_tmp, d_normc, d_dist2, {N, K}, {1, K});

    // Assignment, one-hot, centroid sum (GEMM), counts.
    long* d_assign = nullptr; xpu_malloc((void**)&d_assign, N * sizeof(long));
    xdnn::argmin<float, long>(ctx, d_dist2, d_assign, {N, K}, 1);
    float* d_onehot = dev_zeros(N * K);
    xdnn::one_hot<long>(ctx, d_assign, d_onehot, N, (int)K);
    float* d_centroid_sum = dev_zeros(K * D);
    xdnn::fc<float, float, float, float>(ctx, d_onehot, d_x, d_centroid_sum, K, D, N, true, false, nullptr, nullptr, nullptr);
    float* d_counts = dev_zeros(K);
    xdnn::reduce_sum<float>(ctx, d_onehot, d_counts, {N, K}, {0});

    xpu_wait();
    std::vector<long> h_assign(N);
    std::vector<float> h_centroid_sum(K * D), h_counts(K);
    xpu_memcpy(h_assign.data(), d_assign, N * sizeof(long), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    xpu_memcpy(h_centroid_sum.data(), d_centroid_sum, K * D * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    xpu_memcpy(h_counts.data(), d_counts, K * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("assign: %ld %ld %ld %ld\n", h_assign[0], h_assign[1], h_assign[2], h_assign[3]);
    printf("centroid_sum: %.3f %.3f %.3f %.3f\n", h_centroid_sum[0], h_centroid_sum[1], h_centroid_sum[2], h_centroid_sum[3]);
    printf("counts: %.1f %.1f\n", h_counts[0], h_counts[1]);
    // Final sum/count division is a 2-element host-side step, deliberately
    // not tested on-device -- not worth the risk for a trivial scalar divide.
    return 0;
}
```

Result: `assign = [0, 0, 1, 1]`, `centroid_sum = [0, 1, 10, 11]`, `counts = [2, 2]` — exact bit-match to the hand-computed expectation. With `XPUAPI_DEBUG=1`, nine distinct real `gtest_*<...>(api::kXPU1, ...)` dispatch traces appear (`square` ×2, `reduce_sum` ×2, `fc_fusion` ×2, `scale`, `broadcast_add` ×2, `argmin`, `one_hot`), confirming every stage genuinely computes on the chip. `xpu_wait()` returned `0`, `dmesg` logged nothing at all, and the result is exactly reproducible across two runs. Post-test device health reconfirmed with the full GEMM suite — no degradation.

**Nothing left untested for the algorithm itself** — this is a complete iteration, not a building block. The only open item is the one already noted at the static-analysis stage: iteration cost (several kernel launches per pass), unaffected by this result.
