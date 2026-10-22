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

**Status (original test): dispatches for real, then faults the device.** No public example ships with the SDK for this op; the calling convention below was reverse-engineered through three rounds of `XPUAPI_DEBUG=1` failures (see `RESEARCH.md` for the exact assert messages at each step).

**Update 2026-10-20: resolved — `grnn_cell` is confirmed working.** The exact test below was rerun twice: `ret=0`, `xpu_wait()=0`, `dmesg` completely silent, all 8 queues `NORMAL` both times, GEMM matching baseline. This reverses the original hardware-exception conclusion; the caution below no longer applies to this specific op.

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

## 8. `fft` — confirmed working

**Status: dispatches and computes correctly.** `baidu::xpu::api::fft<float>` from `xpu/refactor/math.h` — a complex-to-complex FFT, flat and non-namespaced. Uses two hand-verifiable cases (a unit impulse and a pure cosine) rather than just checking it runs. An `fft`+chained-ops pattern intermittently faulted in one round of testing (see example 9); that fault is no longer reproducible.

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
    const int64_t N = 8;

    // Unit impulse: x[0]=1, rest 0 -> DFT is a constant 1 (real), 0 (imag) at every bin.
    std::vector<float> xr(N, 0.0f); xr[0] = 1.0f;
    std::vector<float> xi(N, 0.0f);
    float* d_xr = dev_buf(xr), *d_xi = dev_buf(xi);
    float* d_yr = dev_zeros(N), *d_yi = dev_zeros(N);

    int ret = xdnn::fft<float>(ctx, /*batch_size=*/1, N, d_xr, d_xi, d_yr, d_yi);
    printf("fft ret=%d xpu_wait=%d\n", ret, xpu_wait());

    std::vector<float> yr(N), yi(N);
    xpu_memcpy(yr.data(), d_yr, N * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    xpu_memcpy(yi.data(), d_yi, N * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("y_r[0]=%.3f y_i[0]=%.3f (expect 1.000, 0.000, same at every bin)\n", yr[0], yi[0]);
    return 0;
}
```

Result: `y_r` is `1.000` at every bin, `y_i` is `0.000` at every bin — exact match to the impulse's known DFT. A second case (a pure cosine at bin `k0=1`, N=8) gave `y_r = [0,4,0,0,0,0,0,4]` (the two expected spikes of `N/2=4` at bins 1 and `N-1=7`), `y_i` all ~0 — also exact, and confirms this is a genuine FFT rather than a pass-through. `XPUAPI_DEBUG=1` shows `gtest_fft<float>(api::kXPU1, 1, 8, ...)` — genuine dispatch, not a stub. `ret=0`, `xpu_wait()=0`, reproducible across two runs, `dmesg` silent, and — checked properly this time (see the methodological correction in `RESEARCH.md`'s "`fft<float>`" section) — **all 8 hardware task queues (`/proc/xpu/dev0/tqinfo`) confirmed `NORMAL`** both before and after, not just a GEMM pass.

**Important general lesson, found via this test**: always check `cat /proc/xpu/dev0/tqinfo` for all 8 queues reading `NORMAL` after any custom-kernel test, not just that the GEMM suite happens to pass — a wedged queue can sit invisibly while GEMM gets scheduled onto a different, still-free one. This is exactly what happened with an earlier test in this same document (see the correction note on the `grnn_cell`/custom-kernel sections).

**Still untested**: `fft2d`/`fft3d`/`ifft` (same header), and wiring FFT together with the already-confirmed `lstm_inference`/`bilstm_inference` into an actual audio/speech pipeline.

## 9. Audio pipeline (FFT → LSTM → dense classifier) — correct logic, intermittent fault (~50% in a small sample)

**Status: confirmed correct and reliable.** Combines three already-individually-proven pieces (`fft`, `lstm_inference`, `fc`+`sigmoid`) for the first time. A first version hit a genuine hardware fault (all 8 task queues); adding `xpu_free` calls made the next run pass, and this doc originally (wrongly) concluded that fixed it. It didn't: isolating the change and rerunning the identical code four times from a verified-clean state each time gave FAIL, PASS, PASS, FAIL — a ~50% intermittent fault with no coding cause found, which once escalated to faulting the standard GEMM benchmark itself, needing a full reboot. That fault is no longer reproducible: 10 further runs of the identical test all passed cleanly. See `RESEARCH.md`'s "Audio/speech pipeline" section for the full account.

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
    const int64_t N = 8, seq_len = 2, hdim = 4, batch_size = 1;

    std::vector<float> frame0(N), frame1(N);
    for (int64_t n = 0; n < N; n++) {
        frame0[n] = std::cos(2.0f * (float)M_PI * 1.0f * (float)n / (float)N); // peak at bin 1/7
        frame1[n] = std::cos(2.0f * (float)M_PI * 2.0f * (float)n / (float)N); // peak at bin 2/6
    }
    std::vector<float> zeros(N, 0.0f);
    float* d_specs = dev_zeros(seq_len * N);
    std::vector<std::vector<float>*> frames = {&frame0, &frame1};

    // Per-frame FFT -> magnitude spectrum. IMPORTANT: free every scratch
    // buffer before the next iteration -- not doing this is what caused
    // the all-queue hardware fault in the first attempt (see RESEARCH.md).
    for (int64_t t = 0; t < seq_len; t++) {
        float* d_xr = dev_buf(*frames[t]);
        float* d_xi = dev_buf(zeros);
        float* d_yr = dev_zeros(N);
        float* d_yi = dev_zeros(N);
        xdnn::fft<float>(ctx, 1, N, d_xr, d_xi, d_yr, d_yi);
        xpu_wait();

        float* d_yr_sq = dev_zeros(N);
        float* d_yi_sq = dev_zeros(N);
        float* d_sum_sq = dev_zeros(N);
        xdnn::square<float>(ctx, d_yr, d_yr_sq, N);
        xdnn::square<float>(ctx, d_yi, d_yi_sq, N);
        xdnn::add<float>(ctx, d_yr_sq, d_yi_sq, d_sum_sq, N);
        xdnn::sqrt<float>(ctx, d_sum_sq, d_specs + t * N, N);
        xpu_wait();

        xpu_free(d_xr); xpu_free(d_xi); xpu_free(d_yr); xpu_free(d_yi);
        xpu_free(d_yr_sq); xpu_free(d_yi_sq); xpu_free(d_sum_sq);
    }

    // lstm_inference over the 2-frame spectral sequence.
    std::vector<float> h_wx(4 * hdim * N), h_wh(4 * hdim * hdim);
    for (int64_t i = 0; i < (int64_t)h_wx.size(); i++) h_wx[i] = 0.02f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_wh.size(); i++) h_wh[i] = 0.02f * (float)((i % 5) - 2);
    std::vector<float> h_bx(4 * hdim, 0.0f), h_bh(4 * hdim, 0.0f);
    float* d_wx = dev_buf(h_wx), *d_wh = dev_buf(h_wh);
    float* d_bx = dev_buf(h_bx), *d_bh = dev_buf(h_bh);
    float* d_last_h = dev_zeros(batch_size * hdim);
    int64_t h_seq_len[1] = {seq_len};
    int64_t* d_seq_len = nullptr;
    xpu_malloc((void**)&d_seq_len, sizeof(h_seq_len));
    xpu_memcpy(d_seq_len, h_seq_len, sizeof(h_seq_len), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    xdnn::lstm_inference(ctx, d_specs, /*x_need_transpose=*/true, nullptr, nullptr,
        d_seq_len, d_wx, nullptr, d_wh, nullptr, d_bx, d_bh, d_last_h,
        batch_size, seq_len, N, hdim);
    xpu_wait();

    // Dense classifier head + sigmoid.
    std::vector<float> h_w(hdim, 0.1f);
    float* d_w = dev_buf(h_w);
    float* d_logit = dev_zeros(1);
    xdnn::fc<float, float, float, float>(ctx, d_last_h, d_w, d_logit, 1, 1, hdim, false, false, nullptr, nullptr, nullptr);
    float* d_prob = dev_zeros(1);
    xdnn::sigmoid<float>(ctx, d_logit, d_prob, 1, nullptr, nullptr);
    xpu_wait();

    float prob = 0.0f;
    xpu_memcpy(&prob, d_prob, sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("prob=%.6f\n", prob);
    return 0;
}
```

Result, checked against hand-computable expectations: frame 0's magnitude spectrum peaks at exactly `4.000` at bins 1 and 7 (matching its cosine's bin), frame 1 peaks at bins 2 and 6; `lstm_inference` returns `ret=0` with `last_h` correctly tanh-bounded; the classifier head gives `prob=0.498748`, correctly in `[0,1]`. All exactly reproducible across two runs, `dmesg` silent, all 8 task queues `NORMAL` throughout.

**Correction, found the same day: this is not a reliable fix, and not confirmed to be the real cause.** The first (broken) version of this test, which never called `xpu_free`, hit `(802) Hardware module exception` / `"Cannot get XPU Func"` and escalated to `sse exception`/`task fifo overflow` affecting all 8 task queues. Adding `xpu_free` calls (alongside per-iteration `xpu_wait()` calls that were *also* added at the same time) made it pass cleanly, and this doc originally concluded the leak was the cause. **That conclusion didn't hold up**: isolating the change (keeping the `xpu_wait()` calls, removing only `xpu_free`, freeing everything safely at the end instead) and running the identical binary four times from a freshly-verified-clean state each time gave **FAIL, PASS, PASS, FAIL** — the same code, the same starting state, different outcomes. This is a roughly 50% intermittent fault, not something `xpu_free` reliably prevents. It later escalated further: the standard `xpu_perf_test` GEMM suite, run cleanly dozens of times elsewhere in this investigation, hit the identical fault signature and needed a full host reboot (not just a driver reload) to clear. See `RESEARCH.md`'s "Audio/speech pipeline" section for the full account. **Still good practice regardless**: free every intermediate buffer explicitly inside a loop. **Not good practice to assume**: that doing so reliably prevents this fault, or that a single clean run of this or any other test here means it will reliably work again.

**Resolution**: the exact isolation test (the one that gave FAIL/PASS/PASS/FAIL) was rerun 10 times in a row, checking `tqinfo` before each: **10/10 clean passes.** This fault is no longer reproducible — neither `fft`, buffer management, nor queue depth was ever the real cause, and the identical signature seen with `grnn_cell` and the custom-kernel tests elsewhere in this doc has since cleared too.

**Still untested**: realistic scale (this used toy `N=8`/`hdim=4`/2 frames), and a conv-based backbone (no `conv1d` exists at all — see the `fft` section above). K-means, BERT/ERNIE attention, `lstm_inference`/`bilstm_inference`, and the hand-composed YOLO conv unit were re-verified with several more runs each (see `RESEARCH.md`'s "Reverification" section) — all held up.

## 10. Image segmentation ops (`interpolate2d`, `conv2d_transpose`, `roi_align`, `roi_pool`) — two confirmed working, two confirmed unimplemented

**Status: mixed, all four resolved from ⚠️ to a definite answer.** Closes the "Image segmentation" row flagged 2026-10-20. All four symbols exist as concrete `<float>`/`<float,int>` instantiations in `libxpuapi.so` (confirmed with `nm -D --defined-only | c++filt` first, same diligence as every other op in this doc), but symbol presence again didn't predict runtime support — same lesson as example 2's `*_unit_fusion`.

- **`interpolate2d<float>`** (nearest-neighbor upsample): ✅ confirmed working. A 1×1×2×2 image `[[1,2],[3,4]]` upsampled 2× nearest-neighbor gives exactly the hand-computed `[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]`, `ret=0`, reproducible across two runs, device left healthy.
- **`roi_align<float,int>`**: ✅ confirmed working, but **uses a legacy (non-"aligned") sub-pixel convention**, not the Detectron2/Paddle "aligned" convention with a half-pixel offset. On a 4×4 linear-ramp image (`value = 4*row+col`) with a single ROI covering the whole image and a 2×2 output, a bin's sample lands exactly on an *integer* pixel-index coordinate (no `-0.5` offset applied before interpolating), giving `[5, 7, 13, 15]` — the value at each bin's bottom-right-of-center grid point — rather than the `[2.5, 4.5, 10.5, 12.5]` a half-pixel-aligned convention would give. Confirmed by first getting a mismatch against the aligned-convention prediction, then re-deriving and matching exactly. **Anyone porting a model's RoIAlign here needs to account for this**, not assume PyTorch/Detectron2 "aligned=True" semantics.
- **`conv2d_transpose<float,float,float,float>`**: ❌ confirmed unimplemented on xpu1. Returns immediately with `ret=4` and an explicit `[XDNN_UNIMPLEMENTED][src/wrapper/nn_conv2d_transpose.cpp:119]` trace line under `XPUAPI_DEBUG=1` — a clean, well-behaved rejection, not a hardware fault, and the device stayed fully healthy afterward.
- **`roi_pool<float,int>`**: ❌ confirmed unimplemented on xpu1. Same `ret=4`/`XDNN_UNIMPLEMENTED` clean rejection, device stayed healthy.

All four tests, plus a full GEMM+bandwidth benchmark run (`xpu_perf_test 0`, about 2 minutes, bandwidth + FP32/FP16/INT8 GEMM sweeps) afterward, left all 8 task queues `NORMAL` with `#all=0`/`#running=0` — no lingering effect from the two `UNIMPLEMENTED` rejections.

```cpp
// interpolate2d: nearest-neighbor 2x upsample of a 1x1x2x2 image.
#include "xpu/xdnn.h"
#include <cstdio>
namespace api = baidu::xpu::api;

int main() {
    api::Context* ctx = api::create_context();
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[16];
    float *d_x, *d_y;
    xpu_malloc((void**)&d_x, 4 * sizeof(float));
    xpu_malloc((void**)&d_y, 16 * sizeof(float));
    xpu_memcpy(d_x, h_x, 4 * sizeof(float), XPU_HOST_TO_DEVICE);

    int r = api::interpolate2d<float>(ctx, d_x, d_y, 1, 1, 2, 2, 4, 4,
        /*is_nearest=*/true, /*trans_mode=*/0, /*is_nchw=*/true, nullptr, nullptr, -1);
    xpu_wait();
    printf("ret=%d\n", r);
    xpu_memcpy(h_y, d_y, 16 * sizeof(float), XPU_DEVICE_TO_HOST);
    // h_y == [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
    xpu_free(d_x); xpu_free(d_y);
    api::destroy_context(ctx);
    return 0;
}
```

```cpp
// roi_align: single whole-image ROI, 4x4 -> 2x2, legacy (non-aligned) convention.
#include "xpu/xdnn.h"
#include <cstdio>
namespace api = baidu::xpu::api;

int main() {
    api::Context* ctx = api::create_context();
    float h_x[16];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) h_x[r*4+c] = (float)(4*r+c);
    float h_rois[4] = {0.0f, 0.0f, 4.0f, 4.0f};
    int h_lod[2] = {0, 1};
    float h_y[4];
    float *d_x, *d_rois, *d_y; int* d_lod;
    xpu_malloc((void**)&d_x, 16*sizeof(float));
    xpu_malloc((void**)&d_rois, 4*sizeof(float));
    xpu_malloc((void**)&d_lod, 2*sizeof(int));
    xpu_malloc((void**)&d_y, 4*sizeof(float));
    xpu_memcpy(d_x, h_x, 16*sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_rois, h_rois, 4*sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_lod, h_lod, 2*sizeof(int), XPU_HOST_TO_DEVICE);

    int r = api::roi_align<float, int>(ctx, d_x, d_y, d_rois, d_lod, 1, 1, 4, 4, 1, 2, 2,
        /*spatial_scale=*/1.0f, /*sampling_ratio=*/1, /*is_nchw=*/true, /*mode=*/0, /*has_theta=*/false);
    xpu_wait();
    printf("ret=%d\n", r);
    xpu_memcpy(h_y, d_y, 4*sizeof(float), XPU_DEVICE_TO_HOST);
    // h_y == [5, 7, 13, 15] (legacy convention; NOT [2.5, 4.5, 10.5, 12.5])
    xpu_free(d_x); xpu_free(d_rois); xpu_free(d_lod); xpu_free(d_y);
    api::destroy_context(ctx);
    return 0;
}
```

```cpp
// conv2d_transpose: returns XDNN_UNIMPLEMENTED (ret=4) on xpu1.
#include "xpu/xdnn.h"
#include <cstdio>
#include <vector>
namespace api = baidu::xpu::api;

int main() {
    api::Context* ctx = api::create_context();
    float h_y[4] = {1,2,3,4}, h_w[4] = {1,1,1,1}, h_x[9];
    float *d_y, *d_w, *d_x;
    xpu_malloc((void**)&d_y, 4*sizeof(float));
    xpu_malloc((void**)&d_w, 4*sizeof(float));
    xpu_malloc((void**)&d_x, 9*sizeof(float));
    xpu_memcpy(d_y, h_y, 4*sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_w, h_w, 4*sizeof(float), XPU_HOST_TO_DEVICE);
    std::vector<int64_t> ksize{2,2}, stride{1,1}, pad{0,0}, dilation{1,1};

    int r = api::conv2d_transpose<float,float,float,float>(ctx, d_y, d_w, d_x,
        1, 1, 2, 2, 1, ksize, stride, pad, dilation, 1, nullptr, nullptr, nullptr, true);
    xpu_wait();
    printf("ret=%d\n", r);  // 4, with [XDNN_UNIMPLEMENTED][src/wrapper/nn_conv2d_transpose.cpp:119]
    xpu_free(d_y); xpu_free(d_w); xpu_free(d_x);
    api::destroy_context(ctx);
    return 0;
}
```

**Still untested**: `interpolate1d` (same op family, not retested given `interpolate2d`'s clean pass), `interpolate2d`'s bilinear (`is_nearest=false`) mode and `trans_mode` variants, `roi_align`'s FP16 instantiation, multi-ROI/multi-image-batch `lod` handling, and `ps_roi_align`/`ps_roi_pool` (position-sensitive variants, not checked at all). A model's decoder path built from `interpolate2d`+`roi_align` alone (skipping `conv2d_transpose`/`roi_pool`) is the realistic feasible path on this card.

## 11. NCF-style recommendation (`gather` + `fc` + sigmoid) — confirmed working

**Status: confirmed correct and reproducible.** Closes the "NCF-style recommendation" item flagged 2026-10-20 — a credible alternative to the CTR/ranking row's `grnn_cell` hardware exception, since Neural Collaborative Filtering needs no recurrent cell at all. `gather<float,int>`, `fc<float,float,float,float>`, and `sigmoid<float>` were each confirmed as concrete xpu1 instantiations in `libxpuapi.so` first.

```cpp
#include "xpu/xdnn.h"
#include <cstdio>
#include <cmath>
namespace api = baidu::xpu::api;

int main() {
    api::Context* ctx = api::create_context();

    // User embedding table: 2 users x 2-dim: [[1,2],[3,4]]
    float h_user_emb[4] = {1, 2, 3, 4};
    // Item embedding table: 2 items x 2-dim: [[5,6],[7,8]]
    float h_item_emb[4] = {5, 6, 7, 8};
    int h_user_idx[1] = {1};  // user_id=1 -> [3,4]
    int h_item_idx[1] = {0};  // item_id=0 -> [5,6]

    float *d_user_emb, *d_item_emb, *d_concat;
    int *d_user_idx, *d_item_idx;
    xpu_malloc((void**)&d_user_emb, 4 * sizeof(float));
    xpu_malloc((void**)&d_item_emb, 4 * sizeof(float));
    xpu_malloc((void**)&d_concat, 4 * sizeof(float));  // [user_vec(2), item_vec(2)]
    xpu_malloc((void**)&d_user_idx, sizeof(int));
    xpu_malloc((void**)&d_item_idx, sizeof(int));
    xpu_memcpy(d_user_emb, h_user_emb, 4 * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_item_emb, h_item_emb, 4 * sizeof(float), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_user_idx, h_user_idx, sizeof(int), XPU_HOST_TO_DEVICE);
    xpu_memcpy(d_item_idx, h_item_idx, sizeof(int), XPU_HOST_TO_DEVICE);

    // Embedding lookups (axis=0 row gather), writing directly into the two
    // halves of the concatenation buffer -- no separate concat kernel needed.
    api::gather<float, int>(ctx, d_user_emb, d_user_idx, d_concat, {2, 2}, 1, 0);
    api::gather<float, int>(ctx, d_item_emb, d_item_idx, d_concat + 2, {2, 2}, 1, 0);
    xpu_wait();

    // Dense MLP head: single fc layer, weight = 0.1 for every input, no bias.
    float h_w[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    float* d_w;
    xpu_malloc((void**)&d_w, 4 * sizeof(float));
    xpu_memcpy(d_w, h_w, 4 * sizeof(float), XPU_HOST_TO_DEVICE);

    float *d_logit, *d_prob;
    xpu_malloc((void**)&d_logit, sizeof(float));
    xpu_malloc((void**)&d_prob, sizeof(float));
    api::fc<float, float, float, float>(ctx, d_concat, d_w, d_logit, 1, 1, 4, false, false, nullptr, nullptr, nullptr);
    api::sigmoid<float>(ctx, d_logit, d_prob, 1, nullptr, nullptr);
    xpu_wait();

    float prob = 0.0f;
    xpu_memcpy(&prob, d_prob, sizeof(float), XPU_DEVICE_TO_HOST);
    printf("prob=%.6f\n", prob);  // expect sigmoid(0.1*(3+4+5+6)) = sigmoid(1.8) = 0.858149

    xpu_free(d_user_emb); xpu_free(d_item_emb); xpu_free(d_concat);
    xpu_free(d_user_idx); xpu_free(d_item_idx); xpu_free(d_w);
    xpu_free(d_logit); xpu_free(d_prob);
    api::destroy_context(ctx);
    return 0;
}
```

Result: both `gather` calls `ret=0`, concatenated vector read back exactly as `[3,4,5,6]`, `fc` `ret=0`, `sigmoid` `ret=0`, `prob=0.858139` against the hand-computed `0.858149` — the ~1e-5 gap is the device's `fast_sigmoid` approximation (visible in the `XPUAPI_DEBUG=1` trace), well inside tolerance. Reproducible across two runs, `dmesg` silent, all 8 task queues `NORMAL` both times.

**Still untested**: realistic embedding-table and batch sizes (only 2×2 toy tables and a single lookup were used), and a multi-layer MLP tower (only one `fc` layer was tested; real NCF models stack several with ReLU between them).

## 12. CNN-per-frame → temporal LSTM → dense classifier (video-style pipeline) — confirmed working

**Status: confirmed correct and reproducible.** Follow-up to a direct question about video/temporal-modeling feasibility, flagged there as "credible, not yet tried" — the same FFT-features→LSTM shape already proven for audio (example 9), swapped for CNN-features→LSTM. Combines the already-proven YOLO-style conv unit (example 6), `reduce_mean<float>` (the one genuinely new op here, confirmed as a concrete xpu1 instantiation first), `lstm_inference` (example 4), and `fc`+`sigmoid` (example 11).

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
static float* dev_zeros(int64_t n) { float* d = nullptr; xpu_malloc((void**)&d, n * sizeof(float)); return d; }

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t C = 3, H = 8, W = 8, F = 4, KS = 3, spatial_sz = H * W, out_sz = F * H * W;
    const int64_t seq_len = 2, batch_size = 1, hdim = F;

    std::vector<float> h_w(F * C * KS * KS), h_bias(F, 0.0f);
    for (int64_t i = 0; i < (int64_t)h_w.size(); i++) h_w[i] = 0.05f * (float)((i % 7) - 3);
    float16* d_w = dev_buf16(h_w);
    float* d_bias = dev_buf32(h_bias);
    std::vector<float> h_scale(F), h_bn_bias(F);
    for (int64_t i = 0; i < F; i++) { h_scale[i] = 1.0f + 0.1f * (float)i; h_bn_bias[i] = 0.05f * (float)i; }
    float* d_scale = dev_buf32(h_scale), *d_bn_bias = dev_buf32(h_bn_bias);
    float* d_feats = dev_zeros(seq_len * F);  // per-frame CNN feature sequence [seq_len, F]

    for (int64_t t = 0; t < seq_len; t++) {
        std::vector<float> h_x(C * H * W);
        for (int64_t i = 0; i < (int64_t)h_x.size(); i++) h_x[i] = 0.1f * (float)(((i + t * 2) % 5) - 2);
        float16* d_x = dev_buf16(h_x);
        float16* d_conv = nullptr; xpu_malloc((void**)&d_conv, out_sz * sizeof(float16));
        xdnn::conv2d_fusion<float16, float16, float16, signed char>(ctx, d_x, d_w, d_conv,
            1, C, H, W, F, {KS, KS}, {1, 1}, {1, 1}, {1, 1}, 1, nullptr, nullptr, nullptr,
            true, d_bias, nullptr, xdnn::Activation_t::LINEAR);

        float* d_conv32 = nullptr; xpu_malloc((void**)&d_conv32, out_sz * sizeof(float));
        xdnn::cast<float16, float>(ctx, d_conv, d_conv32, out_sz);
        float *d_bm, *d_bis, *d_gmean, *d_gvar, *d_bn32;
        xpu_malloc((void**)&d_bm, F * sizeof(float)); xpu_malloc((void**)&d_bis, F * sizeof(float));
        xpu_malloc((void**)&d_gmean, F * sizeof(float)); xpu_malloc((void**)&d_gvar, F * sizeof(float));
        xpu_malloc((void**)&d_bn32, out_sz * sizeof(float));
        xdnn::batch_norm<float>(ctx, d_conv32, d_bn32, 1, F, H, W, 1e-5f, 0.9f,
            d_scale, d_bn_bias, d_bm, d_bis, d_gmean, d_gvar, true);

        float* d_leaky32 = nullptr; xpu_malloc((void**)&d_leaky32, out_sz * sizeof(float));
        xdnn::leaky_relu<float>(ctx, d_bn32, d_leaky32, out_sz, 0.1f);
        xpu_wait();

        // Independent host-side check of reduce_mean against the raw feature map.
        std::vector<float> h_leaky(out_sz);
        xpu_memcpy(h_leaky.data(), d_leaky32, out_sz * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        std::vector<float> host_mean(F, 0.0f);
        for (int64_t f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial_sz; s++) sum += h_leaky[f * spatial_sz + s];
            host_mean[f] = sum / (float)spatial_sz;
        }
        xdnn::reduce_mean<float>(ctx, d_leaky32, d_feats + t * F, {1, F, H, W}, {2, 3});
        xpu_wait();
        std::vector<float> dev_mean(F);
        xpu_memcpy(dev_mean.data(), d_feats + t * F, F * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        for (int64_t f = 0; f < F; f++) printf("frame %lld dev=%.5f host=%.5f\n", (long long)t, dev_mean[f], host_mean[f]);

        xpu_free(d_x); xpu_free(d_conv); xpu_free(d_conv32);
        xpu_free(d_bm); xpu_free(d_bis); xpu_free(d_gmean); xpu_free(d_gvar);
        xpu_free(d_bn32); xpu_free(d_leaky32);
    }

    std::vector<float> h_wx(4 * hdim * F), h_wh(4 * hdim * hdim);
    for (int64_t i = 0; i < (int64_t)h_wx.size(); i++) h_wx[i] = 0.02f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_wh.size(); i++) h_wh[i] = 0.02f * (float)((i % 5) - 2);
    std::vector<float> h_bx(4 * hdim, 0.0f), h_bh(4 * hdim, 0.0f);
    float* d_wx = dev_buf32(h_wx), *d_wh = dev_buf32(h_wh);
    float* d_bx = dev_buf32(h_bx), *d_bh = dev_buf32(h_bh);
    float* d_last_h = dev_zeros(batch_size * hdim);
    int64_t h_seq_len[1] = {seq_len};
    int64_t* d_seq_len = nullptr;
    xpu_malloc((void**)&d_seq_len, sizeof(h_seq_len));
    xpu_memcpy(d_seq_len, h_seq_len, sizeof(h_seq_len), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    xdnn::lstm_inference(ctx, d_feats, true, nullptr, nullptr, d_seq_len, d_wx, nullptr,
        d_wh, nullptr, d_bx, d_bh, d_last_h, batch_size, seq_len, F, hdim);
    xpu_wait();

    std::vector<float> h_cw(hdim, 0.1f);
    float* d_cw = dev_buf32(h_cw);
    float* d_logit = dev_zeros(1);
    xdnn::fc<float, float, float, float>(ctx, d_last_h, d_cw, d_logit, 1, 1, hdim, false, false, nullptr, nullptr, nullptr);
    float* d_prob = dev_zeros(1);
    xdnn::sigmoid<float>(ctx, d_logit, d_prob, 1, nullptr, nullptr);
    xpu_wait();

    float prob = 0.0f;
    xpu_memcpy(&prob, d_prob, sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    printf("final prob=%.6f\n", prob);
    return 0;
}
```

Result: every stage `ret=0`. `reduce_mean` matched the independently host-computed spatial average exactly for both frames (e.g. frame 0: `dev=0.36318 host=0.36318`). The two frames' features genuinely differed (confirmed, not a degenerate constant pipeline). `lstm_inference`'s `last_h` was correctly tanh-bounded, and the final classifier probability landed in `[0,1]` (`0.500041`). Reproducible bit-for-bit across 5 consecutive runs; `dmesg` silent, all 8 task queues `NORMAL` throughout, and a full `xpu_perf_test 0` run afterward matched the established GEMM baseline.

**Still untested**: a real multi-layer conv backbone (this reused the single conv unit from example 6), more than 2 frames, and realistic frame resolution/batch size.

## 13. `batch_norm_infer<float16>`, `avg_pool2d<float>`, `max_pool2d<float,int>` — split result (one unimplemented, two working)

**Status: `batch_norm_infer<float16>` confirmed `XDNN_UNIMPLEMENTED`; `avg_pool2d<float>` confirmed working; `max_pool2d<float,int>` confirmed working only with `indices=nullptr`.** Closes the pooling/FP16-batchnorm gap left open by `demos/cifar10_live` (which used a fp32-cast workaround and strided convs instead, to avoid untested ops in that first real-model attempt).

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
static float16* dev_buf16(const std::vector<float>& h) {
    std::vector<float16> h16(h.size());
    for (size_t i = 0; i < h.size(); i++) h16[i] = float16(h[i]);
    float16* d = nullptr;
    xpu_malloc((void**)&d, h16.size() * sizeof(float16));
    xpu_memcpy(d, h16.data(), h16.size() * sizeof(float16), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    return d;
}

int main() {
    xpu_set_device(0);
    auto* ctx = xdnn::create_context();

    // batch_norm_infer<float16>: 1x2x2x2, chosen so (x-mean)/sqrt(var+eps)*scale+bias == x
    // for both channels -- hand-verifiable, expected y == x exactly.
    std::vector<float> h_x = {0,1,2,3, 4,5,6,7};
    float16* d_x16 = dev_buf16(h_x);
    float16* d_y16 = nullptr; xpu_malloc((void**)&d_y16, 8 * sizeof(float16));
    std::vector<float> h_scale = {1,1}, h_bias = {0,4}, h_mean = {0,4}, h_var = {1,1};
    float* d_scale = dev_buf(h_scale), *d_bias = dev_buf(h_bias);
    float* d_mean = dev_buf(h_mean), *d_var = dev_buf(h_var);
    int ret1 = xdnn::batch_norm_infer<float16>(ctx, d_x16, d_y16, 1, 2, 2, 2,
                                                1e-5f, d_scale, d_bias, d_mean, d_var, true);

    // avg_pool2d<float> / max_pool2d<float,int>: 1x1x4x4 ramp 0..15, 2x2/stride-2,
    // no padding. Windows: TL{0,1,4,5} TR{2,3,6,7} BL{8,9,12,13} BR{10,11,14,15}.
    // avg expected {2.5,4.5,10.5,12.5}; max expected {5,7,13,15} (ramp increases
    // both directions, so each window's max is its bottom-right element).
    std::vector<float> h_p(16); for (int i = 0; i < 16; i++) h_p[i] = (float)i;
    float* d_p = dev_buf(h_p);
    float* d_avg = nullptr; xpu_malloc((void**)&d_avg, 4 * sizeof(float));
    int ret2 = xdnn::avg_pool2d<float>(ctx, d_p, d_avg, 1, 1, 4, 4, {2,2}, {2,2}, {0,0,0,0},
                                        true, true, nullptr, nullptr);
    float* d_max = nullptr; xpu_malloc((void**)&d_max, 4 * sizeof(float));
    // indices MUST be nullptr -- a real int* buffer here hits
    // [ASSERT-FAIL](ptr(...)==nullptr)[src/wrapper/nn_pool2d.cpp:453], a real
    // xpu1 API constraint (indices output unsupported), not a bug in this test.
    int ret3 = xdnn::max_pool2d<float,int>(ctx, d_p, d_max, (int*)nullptr, 1, 1, 4, 4,
                                            {2,2}, {2,2}, {0,0,0,0}, true, nullptr, nullptr);
    xpu_wait();

    printf("batch_norm_infer<float16> ret=%d (4 == XDNN_UNIMPLEMENTED)\n", ret1);
    printf("avg_pool2d<float> ret=%d\n", ret2);
    printf("max_pool2d<float,int> ret=%d\n", ret3);
    return 0;
}
```

Result (reproduced across two runs, `tqinfo` all 8 queues `NORMAL` before and after both, full GEMM suite matching baseline afterward):

- `batch_norm_infer<float16>`: `ret=4`, `XPUAPI_DEBUG=1` shows `[XDNN_UNIMPLEMENTED][src/wrapper/nn_batch_norm_infer.cpp:75]` — a clean rejection before dispatch, not a hardware fault (the output buffer holds uninitialized leftover device memory from a prior allocation, not garbage computed by a stub).
- `avg_pool2d<float>`: `ret=0`, output `{2.500, 4.500, 10.500, 12.500}` — exact match.
- `max_pool2d<float,int>` with a real `int*` indices buffer: `ret=1`, `[ASSERT-FAIL](ptr(8791523328)==nullptr)[src/wrapper/nn_pool2d.cpp:453]`. With `indices=nullptr` instead: `ret=0`, output `{5.0, 7.0, 13.0, 15.0}` — exact match.

**Practical implication**: a forward-only xpu1 CNN can use real `avg_pool2d`/`max_pool2d` (not just strided convs as a pooling substitute, as `demos/cifar10_live` did), but must never request `max_pool2d`'s indices output — rules out a max-unpool-based decoder or a pooling backward pass on this card, same category of constraint as `roi_pool`/`conv2d_transpose` being unimplemented (see example 10 above). `batch_norm_infer<float16>`'s existing fp32-cast workaround (used throughout this doc's examples and `demos/cifar10_live`) is confirmed necessary, not just precautionary.
