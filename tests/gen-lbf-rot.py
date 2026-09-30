#!/usr/bin/env python3
"""Generate tiny synthetic GGUF fixtures for test-lbf-rot.

Implements the lowbitFlash rotation contract (see HANDOFF.md):
  - per-tensor block partition: rot_block_sizes(C), e.g. 2560 -> [1024,1024,512]
  - signs: sha256-derived +/-1 per (tensor name, block index), seed 0xB02A1E
  - weight fold: W_r = rotate_rows(W) = W @ block-diag(D_b H_b), quantized
  - runtime computes y = dequant(W_r) @ R(x), R(x) = concat_b(H_b (D_b x_b))
    so y == (W_r R) x == W x  -> the oracle is dequant(W_r) un-rotated back to
    the primal basis, times the original activation x.

Fixtures embed raw PQ2_0 / PTQ1_0 packed payloads (no ggml quantize pass) so the
test exercises the on-disk byte contract, not the ggml quantizer.

usage:
  python3 gen-lbf-rot.py out.gguf [--flip-signs] [--ptq1]
"""
import argparse, hashlib, math, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gguf-py"))
from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402

GROUP = 128
ROT_SEED = 0xB02A1E
ROT_NOMINAL = 1024

W_NAME = "blk.0.attn_qkv.weight"     # dense rotated weight   [8, 2560]
D_NAME = "blk.0.ffn_down_exps.weight"  # fused-expert weight  [4, 640] (4 experts x [1,640] folded as [4,640] for the mm test)


def rot_block_sizes(C, nominal=ROT_NOMINAL):
    sizes = []
    rem = C
    while rem > 0:
        lp2 = 1 << (rem.bit_length() - 1)
        b = min(nominal, lp2)
        sizes.append(b)
        rem -= b
    assert all(s % GROUP == 0 for s in sizes)
    return sizes


def block_signs(seed, name, blk_idx, n):
    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        d = hashlib.sha256(f"{seed}:{name}:{blk_idx}:{n}:{i // 256}".encode()).digest()
        bits = np.unpackbits(np.frombuffer(d, dtype=np.uint8))
        k = min(256, n - i)
        out[i:i + k] = bits[:k].astype(np.float32) * 2.0 - 1.0
        i += k
    return out


def fwht(x):
    a = np.ascontiguousarray(x, dtype=np.float32)
    n = a.shape[-1]
    assert n & (n - 1) == 0
    h = 1
    shp = a.shape[:-1]
    while h < n:
        a = a.reshape(*shp, n // (2 * h), 2, h)
        u = a[..., 0, :].copy()
        v = a[..., 1, :].copy()
        a[..., 0, :] = u + v
        a[..., 1, :] = u - v
        a = a.reshape(*shp, n)
        h *= 2
    return a * np.float32(1.0 / math.sqrt(n))


def rotate_rows(Wf, name, seed=ROT_SEED):
    """W @ block-diag(D_b H_b): the folded weight basis."""
    R_, C = Wf.shape
    out = np.empty_like(Wf)
    col = 0
    for bi, bs in enumerate(rot_block_sizes(C)):
        D = block_signs(seed, name, bi, bs)
        out[:, col:col + bs] = fwht(Wf[:, col:col + bs] * D)
        col += bs
    return out


def inv_rotate_rows(Wr, name, seed=ROT_SEED):
    """rotated-basis rows -> primal basis: rows @ block-diag(H_b D_b)."""
    R_, C = Wr.shape
    out = np.empty_like(Wr)
    col = 0
    for bi, bs in enumerate(rot_block_sizes(C)):
        D = block_signs(seed, name, bi, bs)
        out[:, col:col + bs] = fwht(Wr[:, col:col + bs]) * D
        col += bs
    return out


def quantize_ternary_g128(Wf):
    """codes u8 = q+1 in {0,1,2}; scales fp16 absmean; returns codes, scales, deq."""
    R, C = Wf.shape
    ng = C // GROUP
    scales = np.empty((R, ng), dtype=np.float16)
    codes = np.empty((R, C), dtype=np.uint8)
    deq = np.empty((R, C), dtype=np.float32)
    for gi in range(ng):
        Wg = Wf[:, gi * GROUP:(gi + 1) * GROUP]
        s32 = np.abs(Wg).mean(axis=1)
        s16 = s32.astype(np.float16)
        s16f = s16.astype(np.float32)
        ss = np.where(s16f > 0, s16f, np.float32(1.0))
        q = np.clip(np.rint(Wg / ss[:, None]), -1.0, 1.0)
        q = np.where(s16f[:, None] > 0, q, np.float32(0.0))
        scales[:, gi] = s16
        codes[:, gi * GROUP:(gi + 1) * GROUP] = (q + 1.0).astype(np.uint8)
        deq[:, gi * GROUP:(gi + 1) * GROUP] = q * s16f[:, None]
    return codes, scales, deq


def pack_pq2_0(codes, scales):
    """PQ2_0 block (34 B / 128 vals): fp16 scale LE + 32 B of 2-bit codes.
    Element j -> byte j//4, bits 2*(j%4); code in {0,1,2} == q+1."""
    R, C = codes.shape
    ng = C // GROUP
    out = np.zeros((R, ng, 34), dtype=np.uint8)
    for g in range(ng):
        sv = scales[:, g].view(np.uint16)
        out[:, g, 0] = (sv & 0xFF).astype(np.uint8)
        out[:, g, 1] = (sv >> 8).astype(np.uint8)
        q = codes[:, g * GROUP:(g + 1) * GROUP].astype(np.uint8)
        for j in range(GROUP):
            byte = j // 4
            sh = (j % 4) * 2
            out[:, g, 2 + byte] |= (q[:, j] << sh)
    return out.reshape(R, ng * 34)


def pack_ptq1_0(codes, scales):
    """PTQ1_0 block (28 B / 128 vals): qs[24] (5 trits/byte over staged strides
    32/16/8) + qh[2] (4 trits/byte over the last 8) + fp16 scale LE.
    Mirrors quantize_row_ptq1_0_ref / dequantize_row_ptq1_0 in ggml-quants.c."""
    R, C = codes.shape
    ng = C // GROUP
    out = np.zeros((R, ng, 28), dtype=np.uint8)
    stages = [32, 16, 8]
    for g in range(ng):
        x = codes[:, g * GROUP:(g + 1) * GROUP].astype(np.int64)  # trits {0,1,2}
        qs = np.zeros((R, 24), dtype=np.uint8)
        j = 0
        base = 0
        for c in stages:
            while j + c <= 24:
                for m in range(c):
                    qv = np.zeros((R,), dtype=np.int64)
                    for n in range(5):
                        qv = qv * 3 + x[:, base + m + n * c]
                    qs[:, j + m] = ((qv * 256 + 242) // 243).astype(np.uint8)
                base += 5 * c
                j += c
        qh = np.zeros((R, 2), dtype=np.uint8)
        for h in range(2):
            qv = np.zeros((R,), dtype=np.int64)
            for m in range(4):
                qv = qv * 3 + x[:, base + h + m * 2]
            qv = qv * 3
            qh[:, h] = ((qv * 256 + 242) // 243).astype(np.uint8)
        out[:, g, :24] = qs
        out[:, g, 24:26] = qh
        sv = scales[:, g].view(np.uint16)
        out[:, g, 26] = (sv & 0xFF).astype(np.uint8)
        out[:, g, 27] = (sv >> 8).astype(np.uint8)
    return out.reshape(R, ng * 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--flip-signs", action="store_true")
    ap.add_argument("--ptq1", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    w = GGUFWriter(args.out, "qwen4exp")

    # ---------------------------------------------------------------- dense
    R1, C1 = 8, 2560
    W1 = rng.standard_normal((R1, C1), dtype=np.float32) * 0.05
    Wr1 = rotate_rows(W1, W_NAME)
    codes1, scales1, deq1_rot = quantize_ternary_g128(Wr1)
    blocks1 = rot_block_sizes(C1)
    signs1 = np.concatenate([block_signs(ROT_SEED, W_NAME, bi, bs)
                             for bi, bs in enumerate(blocks1)]).astype(np.int32)

    # -------------------------------------------- fused-expert (mul_mat_id)
    # weight is [C2, R2E, n_expert] in ggml terms; the fused tensor shares ONE
    # sign set across experts (rotate_rows over the flattened row axis)
    NE, R2E, C2 = 2, 4, 640
    NUSED = 2
    W2 = rng.standard_normal((NE * R2E, C2), dtype=np.float32) * 0.05
    Wr2 = rotate_rows(W2, D_NAME)
    codes2, scales2, deq2_rot = quantize_ternary_g128(Wr2)
    blocks2 = rot_block_sizes(C2)
    signs2 = np.concatenate([block_signs(ROT_SEED, D_NAME, bi, bs)
                             for bi, bs in enumerate(blocks2)]).astype(np.int32)

    if args.flip_signs:
        signs1 = -signs1
        signs2 = -signs2

    w.add_uint32("lowbitflash.rot.version", 1)
    w.add_array("lowbitflash.rot.weight_names", [W_NAME, D_NAME])
    w.add_array("lowbitflash.rot.blocks." + W_NAME, blocks1)
    w.add_array("lowbitflash.rot.signs."  + W_NAME, signs1.tolist())
    w.add_array("lowbitflash.rot.blocks." + D_NAME, blocks2)
    w.add_array("lowbitflash.rot.signs."  + D_NAME, signs2.tolist())

    # tensors: ne[0] = C (input dim), ne[1] = R. GGUF stores ne row-major
    # (ne[0] first); a quantized tensor's data is R rows of C codes each.
    qtype = GGMLQuantizationType.PTQ1_0 if args.ptq1 else GGMLQuantizationType.PQ2_0
    pack = pack_ptq1_0 if args.ptq1 else pack_pq2_0

    # raw_shape is the BYTE shape (packed row last); the writer derives the
    # logical ne[] from it via quant_shape_from_byte_shape
    w.add_tensor(W_NAME, pack(codes1, scales1), raw_shape=[R1, C1 // GROUP * 34 if not args.ptq1 else C1 // GROUP * 28], raw_dtype=qtype)
    # expert e rows are codes2[e*R2E:(e+1)*R2E]; byte shape [NE, R2E, rowbytes]
    rb = C2 // GROUP * (28 if args.ptq1 else 34)
    w.add_tensor(D_NAME, pack(codes2, scales2).reshape(NE, R2E, rb), raw_shape=[NE, R2E, rb], raw_dtype=qtype)

    # activations and the oracle: y = dequant(W_r) @ (R x), where R x applies
    # the same per-block (x*d then H) op the runtime does on activations
    NT = 4
    x1 = rng.standard_normal((C1, NT), dtype=np.float32) * 0.5
    x2 = rng.standard_normal((C2, NUSED * NT), dtype=np.float32) * 0.5
    Rx1 = rotate_rows(x1.T, W_NAME).T          # [C1, NT] activations in rot basis
    ref1 = (deq1_rot @ Rx1).astype(np.float32)

    # mul_mat_id oracle: out[:, u, t] = deq_{ids[u,t]} @ x2[:, u + t*n_used]
    Rx2 = rotate_rows(x2.T, D_NAME).T          # [C2, NUSED*NT]
    deq2u = deq2_rot.reshape(NE, R2E, C2)
    ids = np.array([[0, 1], [1, 0], [0, 1], [1, 1]], dtype=np.int32).T  # [NUSED, NT]
    ref2 = np.zeros((R2E, NUSED, NT), dtype=np.float32)
    for t in range(NT):
        for u in range(NUSED):
            e = int(ids[u, t])
            ref2[:, u, t] = deq2u[e] @ Rx2[:, u + t * NUSED]

    w.add_tensor("x.w",   np.ascontiguousarray(x1.T), raw_shape=[NT, C1])
    x2g = np.ascontiguousarray(x2.reshape(C2, NT, NUSED).transpose(1, 2, 0))  # [NT, NUSED, C2]
    w.add_tensor("x.d",   x2g, raw_shape=[NT, NUSED, C2])
    w.add_tensor("ids.d", np.ascontiguousarray(ids.T), raw_shape=[NT, NUSED])
    w.add_tensor("ref.w", np.ascontiguousarray(ref1.T), raw_shape=[NT, R1])
    w.add_tensor("ref.d", np.ascontiguousarray(ref2.transpose(2, 1, 0)), raw_shape=[NT, NUSED, R2E])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (ptq1={args.ptq1} flip={args.flip_signs})")


if __name__ == "__main__":
    main()
