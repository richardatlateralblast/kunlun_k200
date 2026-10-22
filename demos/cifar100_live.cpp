// Scaled-up successor to cifar10_live: a deeper CNN (4 conv layers vs 3,
// wider channels, real max-pooling instead of strided-conv downsampling)
// trained on a 22-class yard-camera-relevant subset of CIFAR-100 (small
// mammals, insects/invertebrates, flowers/trees, yard-adjacent vehicles --
// see scripts/train_cifar100.py for the exact class list and why).
//
// Architecture (all ops confirmed on xpu1/K100/K200 as of 2026-10-22 --
// see RESEARCH.md and README.md's Capabilities table):
//   conv(3->32,   k3,s1,p1) -> bn_infer -> leaky_relu(0.1)   [32x32x32]
//   max_pool2d(k2,s2, indices=nullptr)                        [16x16x32]
//   conv(32->64,  k3,s1,p1) -> bn_infer -> leaky_relu(0.1)   [16x16x64]
//   max_pool2d(k2,s2, indices=nullptr)                        [8x8x64]
//   conv(64->128, k3,s1,p1) -> bn_infer -> leaky_relu(0.1)   [8x8x128]
//   conv(128->128,k3,s1,p1) -> bn_infer -> leaky_relu(0.1)   [8x8x128]
//   global average pool (reduce_mean)                          [128]
//   fc(128->N) + bias, softmax                                 [N]
//
// Unlike cifar10_live, max_pool2d is real pooling (confirmed working with
// indices=nullptr -- a real device indices buffer hits an internal
// ASSERT-FAIL on xpu1, see RESEARCH.md), not a strided-conv substitute.
// batch_norm_infer<float16> stays XDNN_UNIMPLEMENTED on this chip, so
// normalization still goes through the fp32-cast workaround, same as
// cifar10_live.
//
// Weights come from scripts/train_cifar100.py, loaded from --weights-dir
// at run time (including classes.txt, so the class count/names aren't
// hardcoded here). Input: raw RGB24 frames from stdin, same convention as
// cifar10_live/video_pipeline_live (see scripts/run_cifar100_live.sh).
#include "xpu/xdnn.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
namespace xdnn = baidu::xpu::api;

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

static std::vector<std::string> load_classes(const std::string& dir) {
    std::ifstream f(dir + "/classes.txt");
    if (!f) { fprintf(stderr, "failed to open %s/classes.txt\n", dir.c_str()); exit(1); }
    std::vector<std::string> classes;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) classes.push_back(line);
    }
    return classes;
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
    int64_t in_c, out_c, h, w;  // spatial size unchanged by this block (pad=1,k=3,s=1)
    bool pool_after;             // max_pool2d(k2,s2) after this block
    float16* d_weight;   // [out_c, in_c, 3, 3] fp16
    float* d_bias;       // [out_c] fp32 (conv bias)
    float* d_bn_scale, *d_bn_bias, *d_bn_mean, *d_bn_var;  // [out_c] each
};

static ConvBlock load_block(const std::string& dir, int idx, int64_t in_c, int64_t out_c,
                             int64_t h, int64_t w, bool pool_after) {
    ConvBlock b;
    b.in_c = in_c; b.out_c = out_c; b.h = h; b.w = w; b.pool_after = pool_after;
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

// conv(k3,s1,p1) -> bn_infer -> leaky_relu -> optional max_pool2d(k2,s2).
// Returns the fp16 output buffer and writes the output spatial size to
// out_h/out_w (halved if pool_after). Frees d_x.
static float16* run_block(xdnn::Context* ctx, const ConvBlock& b, float16* d_x,
                           int64_t* out_h, int64_t* out_w) {
    int64_t conv_sz = b.out_c * b.h * b.w;
    float16* d_conv = nullptr; xpu_malloc((void**)&d_conv, conv_sz * sizeof(float16));
    xdnn::conv2d_fusion<float16, float16, float16, signed char>(ctx, d_x, b.d_weight, d_conv,
        1, b.in_c, b.h, b.w, b.out_c, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1,
        nullptr, nullptr, nullptr, true, b.d_bias, nullptr, xdnn::Activation_t::LINEAR);
    xpu_free(d_x);

    float* d_conv32 = nullptr; xpu_malloc((void**)&d_conv32, conv_sz * sizeof(float));
    xdnn::cast<float16, float>(ctx, d_conv, d_conv32, conv_sz);
    xpu_free(d_conv);

    float* d_bn32 = nullptr; xpu_malloc((void**)&d_bn32, conv_sz * sizeof(float));
    xdnn::batch_norm_infer<float>(ctx, d_conv32, d_bn32, 1, b.out_c, b.h, b.w, 1e-5f,
        b.d_bn_scale, b.d_bn_bias, b.d_bn_mean, b.d_bn_var, true);
    xpu_free(d_conv32);

    float* d_leaky32 = nullptr; xpu_malloc((void**)&d_leaky32, conv_sz * sizeof(float));
    xdnn::leaky_relu<float>(ctx, d_bn32, d_leaky32, conv_sz, 0.1f);
    xpu_free(d_bn32);

    if (!b.pool_after) {
        float16* d_out16 = nullptr; xpu_malloc((void**)&d_out16, conv_sz * sizeof(float16));
        xdnn::cast<float, float16>(ctx, d_leaky32, d_out16, conv_sz);
        xpu_free(d_leaky32);
        *out_h = b.h; *out_w = b.w;
        return d_out16;
    }

    // max_pool2d: indices MUST be nullptr -- a real device indices buffer
    // hits an internal ASSERT-FAIL on xpu1 (confirmed 2026-10-22, see
    // RESEARCH.md). Fine here since this is forward-only inference.
    int64_t ph = b.h / 2, pw = b.w / 2;
    float* d_pool32 = nullptr; xpu_malloc((void**)&d_pool32, b.out_c * ph * pw * sizeof(float));
    xdnn::max_pool2d<float,int>(ctx, d_leaky32, d_pool32, (int*)nullptr, 1, b.out_c, b.h, b.w,
        {2,2}, {2,2}, {0,0,0,0}, /*is_nchw=*/true, nullptr, nullptr);
    xpu_free(d_leaky32);

    float16* d_out16 = nullptr; xpu_malloc((void**)&d_out16, b.out_c * ph * pw * sizeof(float16));
    xdnn::cast<float, float16>(ctx, d_pool32, d_out16, b.out_c * ph * pw);
    xpu_free(d_pool32);
    *out_h = ph; *out_w = pw;
    return d_out16;
}

int main(int argc, char** argv) {
    std::string weights_dir = "demos/cifar100_weights";
    int64_t max_frames = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights-dir") && i + 1 < argc) weights_dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoll(argv[++i]);
    }

    std::vector<std::string> classes = load_classes(weights_dir);
    int64_t n_classes = (int64_t)classes.size();

    xpu_set_device(0);
    auto* ctx = xdnn::create_context();

    ConvBlock b1 = load_block(weights_dir, 1, 3, 32, 32, 32, /*pool_after=*/true);
    ConvBlock b2 = load_block(weights_dir, 2, 32, 64, 16, 16, /*pool_after=*/true);
    ConvBlock b3 = load_block(weights_dir, 3, 64, 128, 8, 8, /*pool_after=*/false);
    ConvBlock b4 = load_block(weights_dir, 4, 128, 128, 8, 8, /*pool_after=*/false);
    std::vector<float> h_fcw = load_f32(weights_dir, "fc_weight", n_classes * 128);
    std::vector<float> h_fcb = load_f32(weights_dir, "fc_bias", n_classes);
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

        int64_t h, w;
        float16* d_f1 = run_block(ctx, b1, d_x, &h, &w);
        float16* d_f2 = run_block(ctx, b2, d_f1, &h, &w);
        float16* d_f3 = run_block(ctx, b3, d_f2, &h, &w);
        float16* d_f4 = run_block(ctx, b4, d_f3, &h, &w);

        int64_t feat_sz = b4.out_c * h * w;
        float* d_f4_32 = nullptr; xpu_malloc((void**)&d_f4_32, feat_sz * sizeof(float));
        xdnn::cast<float16, float>(ctx, d_f4, d_f4_32, feat_sz);
        xpu_free(d_f4);
        float* d_gap = nullptr; xpu_malloc((void**)&d_gap, b4.out_c * sizeof(float));
        xdnn::reduce_mean<float>(ctx, d_f4_32, d_gap, {1, b4.out_c, h, w}, {2, 3});
        xpu_free(d_f4_32);

        float* d_logits = nullptr; xpu_malloc((void**)&d_logits, n_classes * sizeof(float));
        xdnn::fc<float, float, float, float>(ctx, d_gap, d_fcw, d_logits, 1, n_classes, 128, false, true, nullptr, nullptr, nullptr);
        xpu_free(d_gap);
        float* d_logits_b = nullptr; xpu_malloc((void**)&d_logits_b, n_classes * sizeof(float));
        xdnn::add<float>(ctx, d_logits, d_fcb, d_logits_b, n_classes);
        xpu_free(d_logits);
        float* d_probs = nullptr; xpu_malloc((void**)&d_probs, n_classes * sizeof(float));
        xdnn::softmax<float>(ctx, d_logits_b, d_probs, {1, n_classes}, 1);
        xpu_free(d_logits_b);
        xpu_wait();

        std::vector<float> h_probs(n_classes);
        xpu_memcpy(h_probs.data(), d_probs, n_classes * sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        xpu_free(d_probs);
        int top = 0;
        for (int64_t i = 1; i < n_classes; i++) if (h_probs[i] > h_probs[top]) top = (int)i;
        printf("frame %lld: %s (%.1f%%) [", (long long)n_frames, classes[top].c_str(), 100.0f * h_probs[top]);
        for (int64_t i = 0; i < n_classes; i++) printf("%s%s=%.3f", i ? " " : "", classes[i].c_str(), h_probs[i]);
        printf("]\n");
    }

    if (n_frames == 0) { fprintf(stderr, "no frames read from stdin\n"); return 1; }
    printf("classified %lld real frame(s)\n", (long long)n_frames);
    xdnn::destroy_context(ctx);
    return 0;
}
