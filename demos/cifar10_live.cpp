// Real recognition against a live camera feed: a small CNN trained on the
// public CIFAR-10 dataset (airplane/automobile/bird/cat/deer/dog/frog/
// horse/ship/truck), classifying real downsampled camera frames.
//
// Architecture (chosen to use ONLY ops already confirmed on xpu1/K100/K200,
// see demos/README.md and RESEARCH.md -- no pooling, no residual/skip
// connections, downsampling via strided conv instead):
//   conv(3->16, k3, s1) -> bn_infer -> leaky_relu(0.1)   [32x32x16]
//   conv(16->32, k3, s2) -> bn_infer -> leaky_relu(0.1)  [16x16x32]
//   conv(32->64, k3, s2) -> bn_infer -> leaky_relu(0.1)  [8x8x64]
//   global average pool (reduce_mean)                    [64]
//   fc(64->10) + bias, softmax                            [10]
//
// Weights come from scripts/train_cifar.py (trained against the real,
// public CIFAR-10 dataset -- see RESEARCH.md), loaded from --weights-dir at
// run time, not embedded in source. Not a sophisticated model (~4-6 conv
// layers, 32x32 input, no attention/skip connections) -- the smallest,
// lowest-hardware-risk step up from the fixed-weight demos, per the scoping
// decision recorded in RESEARCH.md.
//
// Input: raw RGB24 frames from stdin, same convention as video_pipeline_live
// (see scripts/run_cifar10_live.sh), scaled to the trained model's [-1,1]
// normalization.
#include "xpu/xdnn.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
namespace xdnn = baidu::xpu::api;

static const char* CLASSES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};

static std::vector<float> load_f32(const std::string& dir, const std::string& name, size_t expect_n) {
    std::string path = dir + "/" + name + ".f32";
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "failed to open %s\n", path.c_str()); exit(1); }
    std::vector<float> v(expect_n);
    f.read(reinterpret_cast<char*>(v.data()), expect_n * sizeof(float));
    if ((size_t)f.gcount() != expect_n * sizeof(float)) {
        fprintf(stderr, "short read on %s (expected %zu floats)\n", path.c_str(), expect_n);
        exit(1);
    }
    return v;
}

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

struct ConvBlock {
    int64_t in_c, out_c, in_h, in_w, out_h, out_w, stride;
    float16* d_weight;   // [out_c, in_c, 3, 3] fp16
    float* d_bias;       // [out_c] fp32 (conv bias)
    float* d_bn_scale, *d_bn_bias, *d_bn_mean, *d_bn_var;  // [out_c] each
};

static ConvBlock load_block(const std::string& dir, int idx, int64_t in_c, int64_t out_c,
                             int64_t in_h, int64_t in_w, int64_t stride) {
    ConvBlock b;
    b.in_c = in_c; b.out_c = out_c; b.in_h = in_h; b.in_w = in_w; b.stride = stride;
    b.out_h = (in_h + 2 - 3) / stride + 1;  // pad=1, k=3
    b.out_w = (in_w + 2 - 3) / stride + 1;
    std::string p = "conv" + std::to_string(idx);
    b.d_weight = dev_buf16(load_f32(dir, p + "_weight", out_c * in_c * 3 * 3));
    b.d_bias = dev_buf32(load_f32(dir, p + "_bias", out_c));
    std::string bn = "bn" + std::to_string(idx);
    b.d_bn_scale = dev_buf32(load_f32(dir, bn + "_gamma", out_c));
    b.d_bn_bias = dev_buf32(load_f32(dir, bn + "_beta", out_c));
    b.d_bn_mean = dev_buf32(load_f32(dir, bn + "_mean", out_c));
    b.d_bn_var = dev_buf32(load_f32(dir, bn + "_var", out_c));
    return b;
}

// Runs one conv -> bn_infer -> leaky_relu block. Input/output are fp16
// device buffers (fp16 in, fp16 out -- cast to fp32 only around batch_norm,
// which has no fp16-safe inference path proven here).
static float16* run_block(xdnn::Context* ctx, const ConvBlock& b, float16* d_x) {
    int64_t out_sz = b.out_c * b.out_h * b.out_w;
    float16* d_conv = nullptr; xpu_malloc((void**)&d_conv, out_sz * sizeof(float16));
    xdnn::conv2d_fusion<float16, float16, float16, signed char>(ctx, d_x, b.d_weight, d_conv,
        1, b.in_c, b.in_h, b.in_w, b.out_c, {3, 3}, {b.stride, b.stride}, {1, 1}, {1, 1}, 1,
        nullptr, nullptr, nullptr, true, b.d_bias, nullptr, xdnn::Activation_t::LINEAR);

    float* d_conv32 = nullptr; xpu_malloc((void**)&d_conv32, out_sz * sizeof(float));
    xdnn::cast<float16, float>(ctx, d_conv, d_conv32, out_sz);

    float* d_bn32 = nullptr; xpu_malloc((void**)&d_bn32, out_sz * sizeof(float));
    xdnn::batch_norm_infer<float>(ctx, d_conv32, d_bn32, 1, b.out_c, b.out_h, b.out_w, 1e-5f,
        b.d_bn_scale, b.d_bn_bias, b.d_bn_mean, b.d_bn_var, true);

    float* d_leaky32 = nullptr; xpu_malloc((void**)&d_leaky32, out_sz * sizeof(float));
    xdnn::leaky_relu<float>(ctx, d_bn32, d_leaky32, out_sz, 0.1f);

    float16* d_out16 = nullptr; xpu_malloc((void**)&d_out16, out_sz * sizeof(float16));
    xdnn::cast<float, float16>(ctx, d_leaky32, d_out16, out_sz);

    xpu_free(d_conv); xpu_free(d_conv32); xpu_free(d_bn32);
    xpu_free(d_leaky32);
    return d_out16;  // caller frees d_x and this
}

int main(int argc, char** argv) {
    std::string weights_dir = "demos/cifar10_weights";
    int64_t max_frames = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights-dir") && i + 1 < argc) weights_dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoll(argv[++i]);
    }

    xpu_set_device(0);
    auto* ctx = xdnn::create_context();

    ConvBlock b1 = load_block(weights_dir, 1, 3, 16, 32, 32, 1);
    ConvBlock b2 = load_block(weights_dir, 2, 16, 32, b1.out_h, b1.out_w, 2);
    ConvBlock b3 = load_block(weights_dir, 3, 32, 64, b2.out_h, b2.out_w, 2);
    std::vector<float> h_fcw = load_f32(weights_dir, "fc_weight", 10 * 64);
    std::vector<float> h_fcb = load_f32(weights_dir, "fc_bias", 10);
    float* d_fcw = dev_buf32(h_fcw);
    float* d_fcb = dev_buf32(h_fcb);

    const int64_t frame_bytes = 3 * 32 * 32;
    std::vector<unsigned char> raw(frame_bytes);
    int64_t n_frames = 0;
    for (; n_frames < max_frames; n_frames++) {
        size_t got = fread(raw.data(), 1, frame_bytes, stdin);
        if (got != (size_t)frame_bytes) break;

        std::vector<float> h_x(3 * 32 * 32);
        for (int64_t c = 0; c < 3; c++)
            for (int64_t p = 0; p < 32 * 32; p++)
                h_x[c * 32 * 32 + p] = (raw[p * 3 + c] / 255.0f) * 2.0f - 1.0f;
        float16* d_x = dev_buf16(h_x);

        float16* d_f1 = run_block(ctx, b1, d_x); xpu_free(d_x);
        float16* d_f2 = run_block(ctx, b2, d_f1); xpu_free(d_f1);
        float16* d_f3 = run_block(ctx, b3, d_f2); xpu_free(d_f2);

        int64_t feat_sz = b3.out_c * b3.out_h * b3.out_w;
        float* d_f3_32 = nullptr; xpu_malloc((void**)&d_f3_32, feat_sz * sizeof(float));
        xdnn::cast<float16, float>(ctx, d_f3, d_f3_32, feat_sz);
        float* d_gap = nullptr; xpu_malloc((void**)&d_gap, b3.out_c * sizeof(float));
        xdnn::reduce_mean<float>(ctx, d_f3_32, d_gap, {1, b3.out_c, b3.out_h, b3.out_w}, {2, 3});

        float* d_logits = nullptr; xpu_malloc((void**)&d_logits, 10 * sizeof(float));
        xdnn::fc<float, float, float, float>(ctx, d_gap, d_fcw, d_logits, 1, 10, 64, false, true, nullptr, nullptr, nullptr);
        float* d_logits_b = nullptr; xpu_malloc((void**)&d_logits_b, 10 * sizeof(float));
        xdnn::add<float>(ctx, d_logits, d_fcb, d_logits_b, 10);
        float* d_probs = nullptr; xpu_malloc((void**)&d_probs, 10 * sizeof(float));
        xdnn::softmax<float>(ctx, d_logits_b, d_probs, {1, 10}, 1);
        xpu_wait();

        float h_probs[10];
        xpu_memcpy(h_probs, d_probs, 10 * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        int top = 0;
        for (int i = 1; i < 10; i++) if (h_probs[i] > h_probs[top]) top = i;
        printf("frame %lld: %s (%.1f%%) [", (long long)n_frames, CLASSES[top], 100.0f * h_probs[top]);
        for (int i = 0; i < 10; i++) printf("%s%s=%.3f", i ? " " : "", CLASSES[i], h_probs[i]);
        printf("]\n");

        xpu_free(d_f3); xpu_free(d_f3_32); xpu_free(d_gap);
        xpu_free(d_logits); xpu_free(d_logits_b); xpu_free(d_probs);
    }

    if (n_frames == 0) { fprintf(stderr, "no frames read from stdin\n"); return 1; }
    printf("classified %lld real frame(s)\n", (long long)n_frames);
    xdnn::destroy_context(ctx);
    return 0;
}
