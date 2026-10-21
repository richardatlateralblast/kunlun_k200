// K-means clustering, one full iteration: distance -> assign -> centroid
// update, wired together from plain xdnn ops (no fused/recurrent kernels).
// Ported from docs/research/20261019-xpu1-op-testing-examples.md#7, which
// confirmed this exact pipeline on real K200 hardware: nine real kernel
// dispatches, exact bit-match to the hand-computed expectation below, device
// left fully healthy. See demos/README.md for what this file is and isn't.
//
// 4 points, 2 clusters, deliberately well-separated so the right answer is
// obvious by hand: p0=(0,0), p1=(0,1) near c0=(0,0); p2=(5,5), p3=(5,6) near
// c1=(5,5). Expected: assign=[0,0,1,1], final centroids c0=(0,0.5), c1=(5,5.5).
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

    // Final sum/count divide: a 2-element host-side step (deliberately not
    // pushed on-device in the original probe -- not worth the risk for a
    // trivial scalar divide; see the source write-up).
    std::vector<float> new_centroid(K * D);
    for (int64_t k = 0; k < K; ++k)
        for (int64_t d = 0; d < D; ++d)
            new_centroid[k * D + d] = h_centroid_sum[k * D + d] / h_counts[k];
    printf("new centroids: (%.3f, %.3f) (%.3f, %.3f)\n",
           new_centroid[0], new_centroid[1], new_centroid[2], new_centroid[3]);

    long expect_assign[N] = {0, 0, 1, 1};
    float expect_centroid[K * D] = {0.0f, 0.5f, 5.0f, 5.5f};
    bool ok = true;
    for (int64_t i = 0; i < N; ++i) ok &= (h_assign[i] == expect_assign[i]);
    for (int64_t i = 0; i < K * D; ++i) ok &= (std::fabs(new_centroid[i] - expect_centroid[i]) < 1e-4f);
    printf("%s\n", ok ? "PASS" : "FAIL");

    xpu_free(d_x); xpu_free(d_c); xpu_free(d_x_sq); xpu_free(d_c_sq);
    xpu_free(d_normx); xpu_free(d_normc); xpu_free(d_xc); xpu_free(d_neg2xc);
    xpu_free(d_tmp); xpu_free(d_dist2); xpu_free(d_assign); xpu_free(d_onehot);
    xpu_free(d_centroid_sum); xpu_free(d_counts);
    xdnn::destroy_context(ctx);
    return ok ? 0 : 1;
}
