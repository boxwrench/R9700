// Pure target microbenchmark: 1-token vs N-token single-sequence decode.
//
// Removes the proposer entirely so the marginal cost of evaluating extra
// verification positions can be profiled without MTP-head execution in the
// trace. Both arms advance the same number of sequence positions from the same
// starting KV depth, so per-call times are compared at matched depth.
//
// usage: llama-vbench <model.gguf> <n_prefill> <batch_n> <n_iter> [dev_idx] [n_rs_seq]
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <n_prefill> <batch_n> <n_iter> [dev_idx] [n_rs_seq]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const int n_prefill = atoi(argv[2]);
    const int batch_n   = atoi(argv[3]);
    const int n_iter    = atoi(argv[4]);
    const int dev_idx   = argc > 5 ? atoi(argv[5]) : 1;
    const uint32_t n_rs = argc > 6 ? (uint32_t) atoi(argv[6]) : 2;

    llama_backend_init();

    // pick the requested backend device only, so the second GPU is never used
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0, n = ggml_backend_dev_count(); i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(d);
        }
    }
    if ((int) devs.size() <= dev_idx) {
        fprintf(stderr, "device index %d out of range (%zu GPU devices)\n", dev_idx, devs.size());
        return 1;
    }
    fprintf(stderr, "using device: %s\n", ggml_backend_dev_name(devs[dev_idx]));
    std::vector<ggml_backend_dev_t> use_devs = { devs[dev_idx], nullptr };

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.devices      = use_devs.data();
    mparams.split_mode   = LLAMA_SPLIT_MODE_NONE;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 32768;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 512;
    cparams.n_seq_max   = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k      = GGML_TYPE_F16;
    cparams.type_v      = GGML_TYPE_F16;
    cparams.n_rs_seq    = n_rs;
    cparams.n_threads   = 6;
    cparams.n_threads_batch = 6;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    fprintf(stderr, "n_rs_seq reported by context: %u\n", llama_n_rs_seq(ctx));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // deterministic filler tokens; content is irrelevant to timing
    std::vector<llama_token> toks;
    const int n_warm_tokens = batch_n * 8;
    toks.reserve(n_prefill + batch_n * n_iter + n_warm_tokens + 8);
    for (int i = 0; i < n_prefill + batch_n * n_iter + n_warm_tokens + 8; i++) {
        toks.push_back((llama_token) (1000 + (i * 7919) % std::max(1, n_vocab - 2000)));
    }

    // ---- prefill to the requested KV depth
    llama_batch b = llama_batch_init(std::max(batch_n, 512), 0, 1);
    int pos = 0;
    {
        int i = 0;
        while (i < n_prefill) {
            const int n = std::min(512, n_prefill - i);
            b.n_tokens = n;
            for (int j = 0; j < n; j++) {
                b.token[j] = toks[i + j];
                b.pos[j]   = pos + j;
                b.n_seq_id[j] = 1;
                b.seq_id[j][0] = 0;
                b.logits[j] = (i + n >= n_prefill && j == n - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
            i += n; pos += n;
        }
        llama_synchronize(ctx);
    }
    fprintf(stderr, "prefilled to pos=%d\n", pos);

    // ---- warmup iterations (not timed)
    const int n_warm = 5;
    int tok_i = n_prefill;
    for (int it = 0; it < n_warm; it++) {
        b.n_tokens = batch_n;
        for (int j = 0; j < batch_n; j++) {
            b.token[j] = toks[tok_i + j];
            b.pos[j]   = pos + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = 1;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "warmup decode failed\n"); return 1; }
        llama_synchronize(ctx);
        pos += batch_n; tok_i += batch_n;
    }

    // ---- timed loop
    std::vector<double> ms;
    ms.reserve(n_iter);
    for (int it = 0; it < n_iter; it++) {
        b.n_tokens = batch_n;
        for (int j = 0; j < batch_n; j++) {
            b.token[j] = toks[tok_i + j];
            b.pos[j]   = pos + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = 1;   // all positions produce logits, as in verification
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at it=%d\n", it); return 1; }
        llama_synchronize(ctx);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        pos += batch_n; tok_i += batch_n;
    }

    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double sum = 0; for (double v : ms) sum += v;
    auto pct = [&](double p) { return s[std::min(s.size() - 1, (size_t) (p * s.size()))]; };

    printf("RESULT batch_n=%d n_prefill=%d n_iter=%d n_rs_seq=%u "
           "mean=%.4f median=%.4f p10=%.4f p90=%.4f min=%.4f max=%.4f end_pos=%d\n",
           batch_n, n_prefill, n_iter, n_rs,
           sum / ms.size(), pct(0.5), pct(0.10), pct(0.90), s.front(), s.back(), pos);

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
