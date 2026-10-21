// NCF-style recommendation: embedding gather (user + item) -> concat ->
// dense fc -> sigmoid. No recurrent cell, unlike the GRNN/CTR-ranking idea --
// that's the point being demonstrated (see the source write-up).
// Ported from docs/research/20261019-xpu1-op-testing-examples.md#11, which
// confirmed this on real K200 hardware: reproducible across two runs, device
// left fully healthy both times. See demos/README.md for what this file is
// and isn't.
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
    // expect sigmoid(0.1*(3+4+5+6)) = sigmoid(1.8) = 0.858149; the device's
    // fast_sigmoid approximation lands a few 1e-5 off that, not a bug.
    float expect = 1.0f / (1.0f + std::exp(-1.8f));
    bool ok = std::fabs(prob - expect) < 1e-3f;
    printf("prob=%.6f (expect ~%.6f)\n", prob, expect);
    printf("%s\n", ok ? "PASS" : "FAIL");

    xpu_free(d_user_emb); xpu_free(d_item_emb); xpu_free(d_concat);
    xpu_free(d_user_idx); xpu_free(d_item_idx); xpu_free(d_w);
    xpu_free(d_logit); xpu_free(d_prob);
    api::destroy_context(ctx);
    return ok ? 0 : 1;
}
