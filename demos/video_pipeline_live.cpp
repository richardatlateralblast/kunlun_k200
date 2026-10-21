// CNN-per-frame -> temporal LSTM -> dense classifier, fed with REAL frames
// instead of synthetic ones. Same architecture, dimensions and fixed
// (untrained) weights as the confirmed-working toy pipeline in
// docs/research/20261019-xpu1-op-testing-examples.md#12 -- the only change
// is that each frame comes from real pixel data piped in on stdin, instead
// of a synthetic per-frame fill pattern.
//
// This demonstrates the mechanics (decode -> downscale -> on-device CNN ->
// temporal LSTM -> classifier head, all dispatching real xpu1 kernels on
// real video), not a trained model -- the weights are arbitrary fixed
// values, same as every other demo here, so the probability output is not
// meaningful as a real detector. See demos/README.md.
//
// Reads raw 8x8 RGB24 frames from stdin (24 bytes/pixel-row * 8 = 192 bytes
// per frame, 3 bytes/pixel * 64 pixels = 192 bytes/frame), one frame at a
// time, up to --frames (default 6) or until stdin is exhausted. Intended to
// be fed by ffmpeg, e.g.:
//
//   ffmpeg -loglevel error -rtsp_transport tcp -i "$RTSP_URL" \
//       -vf "fps=1,scale=8:8" -pix_fmt rgb24 -f rawvideo -frames:v 6 - \
//     | ./demos/video_pipeline_live --frames 6
//
// No RTSP URL or credentials are hardcoded anywhere in this file or in the
// repo -- the camera's ffmpeg command is run by the caller, with the URL
// supplied at run time only (see demos/README.md).
#include "xpu/xdnn.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char** argv) {
    int64_t max_frames = 6;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoll(argv[++i]);

    xpu_set_device(0);
    auto* ctx = xdnn::create_context();
    const int64_t C = 3, H = 8, W = 8, F = 4, KS = 3, spatial_sz = H * W, out_sz = F * H * W;
    const int64_t hdim = F;
    const int64_t frame_bytes = C * H * W;

    std::vector<float> h_w(F * C * KS * KS), h_bias(F, 0.0f);
    for (int64_t i = 0; i < (int64_t)h_w.size(); i++) h_w[i] = 0.05f * (float)((i % 7) - 3);
    float16* d_w = dev_buf16(h_w);
    float* d_bias = dev_buf32(h_bias);
    std::vector<float> h_scale(F), h_bn_bias(F);
    for (int64_t i = 0; i < F; i++) { h_scale[i] = 1.0f + 0.1f * (float)i; h_bn_bias[i] = 0.05f * (float)i; }
    float* d_scale = dev_buf32(h_scale), *d_bn_bias = dev_buf32(h_bn_bias);
    float* d_feats = dev_zeros(max_frames * F);  // per-frame CNN feature sequence [seq_len, F]

    std::vector<unsigned char> raw(frame_bytes);
    int64_t seq_len = 0;
    for (; seq_len < max_frames; seq_len++) {
        size_t got = fread(raw.data(), 1, frame_bytes, stdin);
        if (got != (size_t)frame_bytes) break;  // stdin exhausted early -- use frames read so far

        // RGB24 bytes -> CHW float in roughly [-1,1], matching the toy
        // pipeline's input scale (it used 0.1*small-int values).
        std::vector<float> h_x(C * H * W);
        for (int64_t c = 0; c < C; c++)
            for (int64_t p = 0; p < spatial_sz; p++)
                h_x[c * spatial_sz + p] = (raw[p * C + c] / 255.0f) * 2.0f - 1.0f;

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
        xdnn::reduce_mean<float>(ctx, d_leaky32, d_feats + seq_len * F, {1, F, H, W}, {2, 3});
        xpu_wait();

        std::vector<float> dev_mean(F);
        xpu_memcpy(dev_mean.data(), d_feats + seq_len * F, F * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        printf("frame %lld features: %.5f %.5f %.5f %.5f\n", (long long)seq_len,
               dev_mean[0], dev_mean[1], dev_mean[2], dev_mean[3]);

        xpu_free(d_x); xpu_free(d_conv); xpu_free(d_conv32);
        xpu_free(d_bm); xpu_free(d_bis); xpu_free(d_gmean); xpu_free(d_gvar);
        xpu_free(d_bn32); xpu_free(d_leaky32);
    }

    if (seq_len == 0) {
        fprintf(stderr, "no frames read from stdin -- nothing to classify\n");
        return 1;
    }
    printf("read %lld real frame(s) from stdin\n", (long long)seq_len);

    std::vector<float> h_wx(4 * hdim * F), h_wh(4 * hdim * hdim);
    for (int64_t i = 0; i < (int64_t)h_wx.size(); i++) h_wx[i] = 0.02f * (float)((i % 5) - 2);
    for (int64_t i = 0; i < (int64_t)h_wh.size(); i++) h_wh[i] = 0.02f * (float)((i % 5) - 2);
    std::vector<float> h_bx(4 * hdim, 0.0f), h_bh(4 * hdim, 0.0f);
    float* d_wx = dev_buf32(h_wx), *d_wh = dev_buf32(h_wh);
    float* d_bx = dev_buf32(h_bx), *d_bh = dev_buf32(h_bh);
    float* d_last_h = dev_zeros(1 * hdim);
    int64_t h_seq_len[1] = {seq_len};
    int64_t* d_seq_len = nullptr;
    xpu_malloc((void**)&d_seq_len, sizeof(h_seq_len));
    xpu_memcpy(d_seq_len, h_seq_len, sizeof(h_seq_len), XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    xdnn::lstm_inference(ctx, d_feats, true, nullptr, nullptr, d_seq_len, d_wx, nullptr,
        d_wh, nullptr, d_bx, d_bh, d_last_h, 1, seq_len, F, hdim);
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
    printf("sequence classifier output: %.6f (untrained weights -- demonstrates the pipeline, not a real detector)\n", prob);

    xdnn::destroy_context(ctx);
    return 0;
}
