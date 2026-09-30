// test-lbf-rot: unit test for the lowbitflash.rot.* segmented rotation contract.
//
// Loads a synthetic GGUF written by tests/gen-lbf-rot.py containing
//   - lowbitflash.rot.{version,weight_names,blocks.*,signs.*} metadata
//   - a PQ2_0 (or PTQ1_0) rotated weight blk.0.attn_qkv.weight  [2560, 8]
//   - a PQ2_0 (or PTQ1_0) fused-expert weight blk.0.ffn_down_exps.weight [640,4,2]
//   - f32 activations x.w [2560,4], x.d [640,2,4], ids.d [2,4] and references
//     ref.w / ref.d computed in numpy as inv_rotate(dequant(W_r)) @ x.
//
// The test builds the same graph the runtime builds through build_lora_mm /
// build_lora_mm_id: llama_lbf_rot_apply(ctx, x, rotation(w)) then the matmul,
// runs it on every available backend, and compares against the oracle.
//
// usage: test-lbf-rot <gguf> [--expect-fail]
//   --expect-fail: the comparison must NOT match (sign-mismatch negative
//                  control); exits 0 iff the output differs as expected.

#include "../src/llama-graph.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const void * tensor_data(const gguf_context * gctx, const std::vector<uint8_t> & blob,
        const char * name, ggml_type * type, int64_t ne[4], int * n_dims) {
    const int64_t tid = gguf_find_tensor(gctx, name);
    if (tid < 0) {
        fprintf(stderr, "missing tensor %s\n", name);
        exit(1);
    }
    *type = (ggml_type) gguf_get_tensor_type(gctx, tid);
    const int64_t * ne_ptr = gguf_get_tensor_ne(gctx, tid);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) { ne[i] = ne_ptr[i]; }
    *n_dims = GGML_MAX_DIMS;
    while (*n_dims > 1 && ne[*n_dims - 1] == 1) { (*n_dims)--; }
    const size_t off = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
    return blob.data() + off;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [--expect-fail]\n", argv[0]);
        return 2;
    }
    const bool expect_fail = argc > 2 && std::string(argv[2]) == "--expect-fail";

    FILE * f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(sz);
    if (fread(blob.data(), 1, sz, f) != (size_t) sz) { fclose(f); return 2; }
    fclose(f);

    gguf_init_params gip = {}; gip.no_alloc = false;
    gguf_context * gctx = gguf_init_from_file(argv[1], gip);
    if (!gctx) { fprintf(stderr, "gguf_init failed\n"); return 2; }

    ggml_type wt; int64_t ne[4]; int nd;
    const void * wd  = tensor_data(gctx, blob, "blk.0.attn_qkv.weight",      &wt, ne, &nd);
    const int64_t C1 = ne[0], R1 = ne[1];
    const ggml_type wqt = wt;
    const void * wd2 = tensor_data(gctx, blob, "blk.0.ffn_down_exps.weight", &wt, ne, &nd);
    const int64_t C2 = ne[0], R2E = ne[1]; const int NE2 = ne[2];
    const void * xd  = tensor_data(gctx, blob, "x.w",   &wt, ne, &nd); const int NT1 = ne[1];
    const void * xd2 = tensor_data(gctx, blob, "x.d",   &wt, ne, &nd); const int NU = ne[1]; const int NT2d = ne[2];
    const void * idd = tensor_data(gctx, blob, "ids.d", &wt, ne, &nd); const int NUSED = ne[0], NT2 = ne[1];
    const void * rd  = tensor_data(gctx, blob, "ref.w", &wt, ne, &nd);
    const void * rd2 = tensor_data(gctx, blob, "ref.d", &wt, ne, &nd);
    const int64_t X2_N1 = NU; // [C2, n_used, n_tok] flattened order

    const size_t blk_bytes = wqt == GGML_TYPE_PQ2_0 ? 34 : 28;
    const size_t w1_sz = (size_t) R1         * (C1 / 128) * blk_bytes;
    const size_t w2_sz = (size_t) R2E * NE2  * (C2 / 128) * blk_bytes;
    (void) X2_N1; (void) NT2d;

    // ---- enumerate backends (GPU first, then CPU)
    ggml_backend_load_all();
    std::vector<ggml_backend_t> backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) { backends.push_back(b); }
        }
    }
    backends.push_back(ggml_backend_cpu_init());

    int failures = 0;
    for (ggml_backend_t backend : backends) {
        const char * bname = ggml_backend_name(backend);

        // everything lives in one no_alloc context; gallocr assigns buffers
        // for leaves AND intermediates in one shot
        ggml_init_params ip = { 4*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * W1 = ggml_new_tensor_2d(ctx, wqt, C1, R1);
        ggml_tensor * W2 = ggml_new_tensor_3d(ctx, wqt, C2, R2E, NE2);
        ggml_tensor * X1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C1, NT1);
        ggml_tensor * X2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C2, NUSED, NT2);
        ggml_tensor * ID = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NUSED, NT2);
        ggml_set_name(W1, "blk.0.attn_qkv.weight");
        ggml_set_name(W2, "blk.0.ffn_down_exps.weight");

        // parse metadata into segment descriptors (tensors still no_alloc)
        llama_lbf_rotation rot1; std::vector<int32_t> s1;
        llama_lbf_rotation rot2; std::vector<int32_t> s2;
        auto load_meta = [&](const std::string & wname, int64_t C,
                             llama_lbf_rotation & rot, std::vector<int32_t> & sign_out) {
            const std::string bk = "lowbitflash.rot.blocks." + wname;
            const std::string sk = "lowbitflash.rot.signs."  + wname;
            const int64_t kb = gguf_find_key(gctx, bk.c_str());
            const int64_t ks = gguf_find_key(gctx, sk.c_str());
            if (kb < 0 || ks < 0) { fprintf(stderr, "missing keys for %s\n", wname.c_str()); exit(1); }
            const int32_t * pb = (const int32_t *) gguf_get_arr_data(gctx, kb);
            const int32_t * ps = (const int32_t *) gguf_get_arr_data(gctx, ks);
            std::vector<int32_t> blocks(pb, pb + gguf_get_arr_n(gctx, kb));
            sign_out.assign(ps, ps + gguf_get_arr_n(gctx, ks));
            int64_t off = 0;
            for (int32_t bs : blocks) {
                ggml_tensor * H = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bs, bs);
                ggml_tensor * S = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bs);
                rot.segs.push_back({ H, S, off, bs });
                off += bs;
            }
        };
        load_meta("blk.0.attn_qkv.weight",      C1, rot1, s1);
        load_meta("blk.0.ffn_down_exps.weight", C2, rot2, s2);

        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * rx1 = llama_lbf_rot_apply(ctx, X1, rot1, false);
        ggml_tensor * y1  = ggml_mul_mat(ctx, W1, rx1);
        ggml_build_forward_expand(gf, y1);

        ggml_tensor * rx2 = llama_lbf_rot_apply(ctx, X2, rot2, false);
        ggml_tensor * y2  = ggml_mul_mat_id(ctx, W2, rx2, ID);
        ggml_build_forward_expand(gf, y2);

        // allocate everything in ctx on this backend
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "%s: gallocr failed\n", bname);
            return 2;
        }

        // fill leaves
        auto fill_sylvester = [&](llama_lbf_rotation & rot, const std::vector<int32_t> & signs) {
            for (auto & s : rot.segs) {
                const int64_t bs = s.bs;
                std::vector<float> h((size_t) bs * bs), sv(bs);
                const float scale = 1.0f / sqrtf((float) bs);
                for (int64_t r = 0; r < bs; ++r)
                    for (int64_t c = 0; c < bs; ++c) {
                        // reference fwht == natural Sylvester Hadamard
                        uint32_t p = (uint32_t) (r & c);
                        p ^= p >> 16; p ^= p >> 8; p ^= p >> 4; p ^= p >> 2; p ^= p >> 1;
                        h[(size_t) r * bs + c] = (p & 1) ? -scale : scale;
                    }
                for (int64_t i = 0; i < bs; ++i) sv[i] = (float) signs[s.off + i];
                ggml_backend_tensor_set(s.rot,   h.data(),  0, h.size()  * 4);
                ggml_backend_tensor_set(s.signs, sv.data(), 0, sv.size() * 4);
            }
        };
        fill_sylvester(rot1, s1);
        fill_sylvester(rot2, s2);
        ggml_backend_tensor_set(W1, wd,  0, w1_sz);
        ggml_backend_tensor_set(W2, wd2, 0, w2_sz);
        ggml_backend_tensor_set(X1, xd,  0, (size_t) C1 * NT1 * 4);
        ggml_backend_tensor_set(X2, xd2, 0, (size_t) C2 * NUSED * NT2 * 4);
        ggml_backend_tensor_set(ID, idd, 0, (size_t) NUSED * NT2 * 4);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", bname);
            return 2;
        }

        std::vector<float> o1(R1 * NT1), o2(R2E * NUSED * NT2);
        ggml_backend_tensor_get(y1, o1.data(), 0, o1.size() * 4);
        ggml_backend_tensor_get(y2, o2.data(), 0, o2.size() * 4);

        auto check = [&](const std::vector<float> & got, const float * ref, int64_t n, const char * tag) {
            float max_err = 0, ref_max = 0;
            for (int64_t i = 0; i < n; ++i) {
                max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
                ref_max = std::max(ref_max, std::fabs(ref[i]));
            }
            const float rel = ref_max > 0 ? max_err / ref_max : max_err;
            printf("  %-12s %s: max_err=%.6g rel=%.4g\n", bname, tag, (double) max_err, (double) rel);
            // PQ2_0/PTQ1_0 fast paths quantize the ACTIVATION to q8 (Q8_K on
            // CPU vec_dot, per-block int8 on CUDA) before the integer dot, so
            // the oracle comparison carries inherent activation-quant error
            // (~6e-3 rel measured). The sign-flip control lands at rel ~= 2.0,
            // giving >100x separation on both sides of the 2e-2 gate.
            return rel < 2e-2 || max_err < 2e-2;
        };

        const bool ok1 = check(o1, (const float *) rd,  R1 * NT1,        "dense[2560]");
        const bool ok2 = check(o2, (const float *) rd2, R2E * NUSED * NT2, "mm_id[640]");
        const bool ok = ok1 && ok2;
        printf("backend %-12s: %s\n", bname, ok ? "match" : "MISMATCH");
        if (!ok) { failures++; }

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    for (auto b : backends) { ggml_backend_free(b); }
    gguf_free(gctx);

    if (expect_fail) {
        if (failures > 0) {
            printf("negative control: mismatch detected as expected\n");
            return 0;
        }
        printf("negative control: UNEXPECTED match (sign flip changed nothing)\n");
        return 1;
    }
    printf(failures ? "FAILED (%d backend mismatch)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
