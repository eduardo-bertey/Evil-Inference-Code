"""Export a trained checkpoint to a `.cact` blob (needle.cact) - the deployment
weights the C++ / single-mega-kernel inference reads (mmap or linked read-only
section; no parsing).

Format is W4A8: the matmul weights (tied embedding + every attention projection)
are Cactus-Quants INT4; norms, Hadamard diagonals, and gates stay FP16. Weights
are PRE-TRANSPOSED to [out, in] so each output row is contiguous along the
reduction axis (what a GEMV/GEMM stream wants, and the axis quant groups run
along). Tensors are LAYER-MAJOR: embedding, then all of layer 0's tensors, ...,
then final norm - so the kernel's working set for one layer is contiguous.

================================ byte layout ================================
All little-endian. A tiny header, then a NAMELESS tensor directory, then 64-byte
aligned tensor blobs. The architecture geometry (dims, layer/engram/mHC layout)
is NOT stored - the runtime bakes it in (canon_config in needle_model.cpp) - so
neither the blob nor the shipped binary reveals the model structure; tensors are
positional in the fixed canon order below.

HEADER  (u32 tag, u32 num_tensors, u32 codebook_len, u32 kv_window, u32
kv_bits; then codebook_len * f32 - the shared Lloyd-Max unit-sphere codebooks,
cb2[4] | cb3[8] | cb4[16] concatenated, so mixed-precision blobs need one
header). kv_window is the sliding-window width the model was trained with (0 =
size it from the KV budget alone); kv_bits is the KV-cache width it was
post-trained for (8 = int8, else 2/3/4). Both ride in the blob because a model
quantized for one must be RUN at it - leaving either to a runtime flag silently
serves the wrong numerics.

DIRECTORY  (num_tensors records, {REC_SIZE} bytes each, no names):
  u8 dtype, u8 ndim, u16 _pad, u32 shape[4], u64 offset, u64 nbytes,
  u32 group_size, u32 bits.  dtype: 1=FP16, 2=FP32, 3=CQ (bits gives the
  width: 2, 3 or 4), 4=RAW.

TENSOR ORDER (what the runtime indexes by position):
  embedding; per layer [norm_in, q_proj, k_proj, v_proj, q_norm, k_norm,
  gate_proj, out_proj, post_norm, attn_gate, pre_hada, d1, d2, d3]; the 9 mHC
  blocks; per engram site [tables, key_proj, value_proj, taps]; final_norm;
  the optional probe heads (see below); then the RAW tokenizer.

PROBE HEADS (all FP16, appended so pre-head blobs stay loadable). Let
  extra = num_tensors - (through final_norm) - (tokenizer). extra == 0 means no
  heads. Otherwise the tensor right after final_norm is a `heads.manifest` of
  shape (H,), one code per head in canonical order (1 contrastive,
  2 confidence), followed by H fixed-stride triples [probes (P, d_model),
  proj (out, P*d_model), bias (out,)], so H = (extra - 1) / 3.

CQ blob for a logical [out, in] matrix (in padded up to a multiple of group):
first the packed indices (out * in_pad*bits/8 bytes), then the per-group L2
norms (out * in_pad/group  FP16). Indices are packed LSB-first in chunks of 8:
each run of 8 consecutive indices is OR'd into a word at offsets i*bits and
emitted as `bits` little-endian bytes. A chunk is a whole number of bytes, so
this is equivalently one continuous LSB-first bitstream per row (index k
occupies bits [k*bits, (k+1)*bits)), which is what the runtime's extractor
assumes; at bits=4 it is the classic nibble packing (low nibble = even column).
Reconstruct per group:
  w_group = (codebook_bits[idx] * norm) @ H  # H = normalized Walsh-Hadamard(group)

TERNARY (record bits=5, 3 levels) stores four signed 2-bit crumbs per byte. The
crumb codes are 3, 0, 1 for trit indices 0, 1, 2, so sign-extending each 2-bit
field directly produces -1, 0, +1 in the NEON kernel. The codebook is not
in the header: the 3-level Lloyd-Max centroids {-c, 0, +c} / sqrt(group),
c = 1.2240064, are analytic.

RAW "tokenizer" blob (self-contained SentencePiece BPE dump; see RefTokenizer
below for the reference encoder/decoder this specifies):
  header  u32 n_pieces, u32 pad/eos/bos/unk id, u8 add_dummy_prefix,
          u8 byte_fallback, u16 _pad;
  then n_pieces records, id order: f32 score, u8 type
          (0 NORMAL, 1 UNKNOWN, 2 CONTROL, 3 USER_DEFINED, 4 BYTE),
          u16 surface_len, surface_len UTF-8 bytes.
=============================================================================
"""
import os
import struct

import numpy as np

from .quantize import (_bits_for, _cq_codebook_np, _cq_hadamard_np,
                       parse_bits_map, TERNARY_BITS)
from .architecture import TransformerConfig

HEAD_STAGES = ("conf-head", "tool-head")

TAG = 0x05E12A82
ALIGN = 64
FP16, FP32, CQ, RAW = 1, 2, 3, 4
CB_BITS = (2, 3, 4)
TERNARY_RECORD_BITS = 5

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_HDR_FMT = "<IIIII"
_REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(_REC_FMT)


def _geometry(config):
    head_dim = (getattr(config, "attn_dim", 0) or config.d_model) // config.num_heads
    orders = tuple(getattr(config, "engram_orders", (2, 3)))
    heads = getattr(config, "engram_heads", 0) or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    sites = tuple(getattr(config, "engram_layers", (2, 15)))
    return head_dim, orders, heads, sub_dim, sites


ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4


def _align(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _nearest_idx(x, cb):
    flat = x.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos)
    return idx.reshape(x.shape).astype(np.uint8)


def _pack_lsb(idx, bits):
    out, in_pad = idx.shape
    g = idx.reshape(out, in_pad // 8, 8).astype(np.uint64)
    word = np.zeros(g.shape[:-1], np.uint64)
    for i in range(8):
        word |= g[..., i] << (i * bits)
    packed = np.empty(word.shape + (bits,), np.uint8)
    for b in range(bits):
        packed[..., b] = (word >> (8 * b)) & 0xFF
    return packed.reshape(out, in_pad * bits // 8)


def _unpack_lsb(packed, bits, in_pad):
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << (8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> (i * bits)) & mask
    return idx.reshape(out, in_pad)


def _packed_row_bytes(in_pad, bits, group):
    if bits == TERNARY_RECORD_BITS:
        return in_pad * 2 // 8
    return in_pad * bits // 8


def _pack_ternary_crumbs(idx):
    crumbs = np.where(idx == 0, 3, idx - 1).astype(np.uint8)
    return _pack_lsb(crumbs, 2)


def _unpack_ternary_crumbs(packed, in_pad):
    crumbs = _unpack_lsb(packed, 2, in_pad)
    return np.where(crumbs == 3, 0, crumbs + 1).astype(np.uint8)


def _cq_pack(w, bits, group):
    if bits not in CB_BITS and bits != TERNARY_BITS:
        raise ValueError(f"CQ packing supports bits in {CB_BITS} or {TERNARY_BITS}; got bits={bits}")
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    in_pad = wp.shape[1]
    g = wp.reshape(out, in_pad // group, group).astype(np.float32)
    rot = g @ H
    norm = np.sqrt((rot ** 2).sum(-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    idx = _nearest_idx(unit, cb).reshape(out, in_pad)
    packed = _pack_ternary_crumbs(idx) if bits == TERNARY_BITS else _pack_lsb(idx, bits)
    return packed, norm[:, :, 0].astype(np.float16)


def _cq_unpack(packed, norms, out, in_dim, bits, group):
    record_bits = bits
    if bits == TERNARY_RECORD_BITS:
        bits = TERNARY_BITS
    cb = _cq_codebook_np(bits, group)
    H = _cq_hadamard_np(group)
    in_pad = (in_dim + group - 1) // group * group
    if record_bits == TERNARY_RECORD_BITS:
        idx = _unpack_ternary_crumbs(packed, in_pad)
    else:
        idx = _unpack_lsb(packed, bits, in_pad)
    unit = cb[idx].reshape(out, in_pad // group, group)
    rot = unit * norms.reshape(out, in_pad // group, 1).astype(np.float32)
    w = (rot @ H).reshape(out, in_pad)
    return w[:, :in_dim]


class _Tensor:
    __slots__ = ("name", "dtype", "shape", "blob", "group", "bits", "offset")

    def __init__(self, name, dtype, shape, blob, group=0, bits=0):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)
        self.blob, self.group, self.bits, self.offset = blob, group, bits, 0


def _fp16(name, arr):
    arr = np.asarray(arr, np.float16)
    return _Tensor(name, FP16, arr.shape, arr.tobytes())


def _q(name, mat, bits, group):
    if isinstance(bits, tuple):
        bits = _bits_for(name, *bits)
    packed, norms = _cq_pack(np.asarray(mat, np.float32), bits, group)
    rec_bits = TERNARY_RECORD_BITS if bits == TERNARY_BITS else bits
    return _Tensor(name, CQ, mat.shape, packed.tobytes() + norms.tobytes(), group, rec_bits)


def _tensors(params, config, bits, group):
    _, orders, heads, sub_dim, sites = _geometry(config)
    num_tables = len(orders) * heads
    ts = [_q("embedding", np.asarray(_get(params, ("embedding", "embedding"))), bits, group)]

    b = params["stack"]["layers"]["block"]
    sa, ha = b["self_attn"], b["hadamard_mlp"]
    for i in range(config.num_layers):
        ts += [
            _fp16(f"layer{i:02d}.norm_in", b["ZCRMSNorm_0"]["scale"][i]),
            _q(f"layer{i:02d}.q_proj", np.asarray(sa["q_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.k_proj", np.asarray(sa["k_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.v_proj", np.asarray(sa["v_proj"]["kernel"][i]).T, bits, group),
            _fp16(f"layer{i:02d}.q_norm", sa["q_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.k_norm", sa["k_norm"]["scale"][i]),
            _q(f"layer{i:02d}.gate_proj", np.asarray(sa["gate_proj"]["kernel"][i]).T, bits, group),
            _q(f"layer{i:02d}.out_proj", np.asarray(sa["out_proj"]["kernel"][i]).T, bits, group),
            _fp16(f"layer{i:02d}.post_norm", b["post_attn_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.attn_gate", np.asarray(b["attn_gate"][i]).reshape(1)),
            _fp16(f"layer{i:02d}.pre_hada", b["pre_hada_norm"]["scale"][i]),
            _fp16(f"layer{i:02d}.d1", ha["d1"][i]),
            _fp16(f"layer{i:02d}.d2", ha["d2"][i]),
            _fp16(f"layer{i:02d}.d3", ha["d3"][i]),
        ]

    mhc = params["stack"]
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
                 "mhc_b_res"):
        ts.append(_fp16(name, np.asarray(mhc[name])))
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = np.asarray(mhc[name])
        L, nC, lanes = phi.shape
        ts.append(_q(name, phi.transpose(0, 2, 1).reshape(L * lanes, nC), bits, group))

    for s in range(len(sites)):
        eg = params[f"engrams_{s}"]
        tables = np.asarray(eg["embedding"]).reshape(num_tables * config.engram_slots, sub_dim)
        ts += [
            _q(f"engram{s}.tables", tables, bits, group),
            _q(f"engram{s}.key_proj", np.asarray(eg["key_proj"]["kernel"]).T, bits, group),
            _q(f"engram{s}.value_proj", np.asarray(eg["value_proj"]["kernel"]).T, bits, group),
            _fp16(f"engram{s}.taps", np.asarray(eg["taps"])),
        ]

    ts.append(_fp16("final_norm", params["stack"]["final_norm"]["scale"]))
    ts += _head_tensors(params)
    return ts


HEAD_CODES = (("contrastive_head", 1), ("confidence_head", 2))


def _head_tensors(params):
    present = [(k, c) for k, c in HEAD_CODES if k in params]
    if not present:
        return []
    ts = [_fp16("heads.manifest", np.asarray([c for _, c in present], np.float16))]
    for key, _ in present:
        h = params[key]
        kernel = np.asarray(h["proj"]["kernel"]).T
        bias = (np.asarray(h["proj"]["bias"]) if "bias" in h["proj"]
                else np.zeros(kernel.shape[0], np.float16))
        ts += [
            _fp16(f"{key}.probes", np.asarray(h["probes"])),
            _fp16(f"{key}.proj", kernel),
            _fp16(f"{key}.bias", bias),
        ]
    return ts


def _get(params, keys):
    node = params
    for k in keys:
        node = node[k]
    return node


def _tokenizer_blob(tok):
    from .tokenizer import CHAT_MARKERS, PAD_ID, EOS_ID, BOS_ID, UNK_ID
    sp = tok.sp
    n = sp.GetPieceSize()
    markers = set(CHAT_MARKERS)
    out = bytearray(struct.pack(_TK_HDR, n, PAD_ID, EOS_ID, BOS_ID, UNK_ID, 1, 1, 0))
    for i in range(n):
        piece = sp.IdToPiece(i)
        if sp.IsControl(i):
            t = TK_CONTROL
        elif sp.IsUnknown(i):
            t = TK_UNKNOWN
        elif sp.IsByte(i):
            t = TK_BYTE
        elif piece in markers:
            t = TK_USER_DEFINED
        else:
            t = TK_NORMAL
        b = piece.encode("utf-8")
        out += struct.pack(_TK_REC, sp.GetScore(i), t, len(b)) + b
    return bytes(out)


def _pack_cact(params, config, bits, group, tokenizer, kv_window=0):
    params = {k: v for k, v in params.items()}
    ts = _tensors(params, config, bits, group)
    if tokenizer is not None:
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(f"tokenizer vocab {tokenizer.vocab_size} != config {config.vocab_size}")
        ts.append(_Tensor("tokenizer", RAW, (), _tokenizer_blob(tokenizer)))
    cb = np.concatenate([_cq_codebook_np(b, group) for b in CB_BITS]).astype(np.float32)
    kv_bits = int(getattr(config, "kv_bits", 8) or 8)
    header = struct.pack(_HDR_FMT, TAG, len(ts), len(cb),
                         int(kv_window or 0), kv_bits) + cb.tobytes()

    pos = len(header) + len(ts) * REC_SIZE
    for t in ts:
        pos = _align(pos)
        t.offset = pos
        pos += len(t.blob)

    directory = b"".join(
        struct.pack(_REC_FMT, t.dtype, len(t.shape), 0,
                    *(list(t.shape) + [0, 0, 0, 0])[:4], t.offset, len(t.blob),
                    t.group, t.bits)
        for t in ts
    )

    buf = bytearray(header + directory)
    for t in ts:
        buf.extend(b"\x00" * (t.offset - len(buf)))
        buf.extend(t.blob)
    return bytes(buf), len(ts)


def build_export(params, config, bits=4, group=128, tokenizer=None, bits_map=None,
                 kv_window=0):
    b = parse_bits_map(bits_map) if bits_map else bits
    return _pack_cact(params, config, b, group, tokenizer, kv_window)[0]


def write_export(params, config, path, bits=4, group=128, tokenizer=None,
                 bits_map=None, kv_window=0):
    b = parse_bits_map(bits_map) if bits_map else bits
    buf, n = _pack_cact(params, config, b, group, tokenizer, kv_window)
    with open(path, "wb") as f:
        f.write(buf)
    return {"path": path, "bytes": len(buf), "tensors": n}


def read_export(path):
    with open(path, "rb") as f:
        raw = f.read()
    tag, num_tensors, cb_n, kv_window, kv_bits = struct.unpack_from(_HDR_FMT, raw, 0)
    assert tag == TAG, "bad tag"
    off = struct.calcsize(_HDR_FMT)
    codebook = np.frombuffer(raw[off:off + cb_n * 4], np.float32).copy()
    off += cb_n * 4

    tensors = []
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE]); off += REC_SIZE
        dtype, ndim = rec[0], rec[1]
        shape = tuple(rec[3:3 + ndim])
        offset, nbytes, group, bits = rec[7], rec[8], rec[9], rec[10]
        blob = raw[offset:offset + nbytes]
        if dtype == FP16:
            tensors.append(np.frombuffer(blob, np.float16).reshape(shape).astype(np.float32))
        elif dtype == FP32:
            tensors.append(np.frombuffer(blob, np.float32).reshape(shape))
        elif dtype == CQ:
            out, in_dim = shape
            in_pad = (in_dim + group - 1) // group * group
            n_packed = out * _packed_row_bytes(in_pad, bits, group)
            packed = np.frombuffer(blob[:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(blob[n_packed:], np.float16).reshape(out, in_pad // group)
            tensors.append(_cq_unpack(packed, norms, out, in_dim, bits, group))
        elif dtype == RAW:
            tensors.append(blob)
    return {"num_tensors": num_tensors, "codebook": codebook,
            "kv_window": kv_window, "kv_bits": kv_bits}, tensors


_SP_META_SPACE = "▁"


def parse_tokenizer_blob(blob):
    off = struct.calcsize(_TK_HDR)
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
    rec = struct.calcsize(_TK_REC)
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, t, ln = struct.unpack_from(_TK_REC, blob, off)
        off += rec
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    return {"pieces": pieces, "scores": scores, "types": types, "pad_id": pad,
            "eos_id": eos, "bos_id": bos, "unk_id": unk,
            "add_dummy_prefix": bool(add_dummy), "byte_fallback": bool(byte_fb)}


class RefTokenizer:
    def __init__(self, meta):
        self.pieces = meta["pieces"]
        self.scores = meta["scores"]
        self.types = meta["types"]
        self.add_dummy = meta["add_dummy_prefix"]
        self.byte_fallback = meta["byte_fallback"]
        self.unk_id = meta["unk_id"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i
                        for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types) if t == TK_USER_DEFINED),
                              key=len, reverse=True)

    @classmethod
    def from_cact(cls, path):
        _, ts = read_export(path)
        blob = next(t for t in ts if isinstance(t, (bytes, bytearray)))
        return cls(parse_tokenizer_blob(blob))

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best_score, best_j = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best_score is None or self.scores[idx] > best_score):
                    best_score, best_j = self.scores[idx], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf)); buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i]); i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text


def main(args):
    from .run import load_checkpoint
    from .tokenizer import get_tokenizer
    params, config, run_meta = load_checkpoint(args.checkpoint, return_run=True)
    out = getattr(args, "out", None) or "needle.cact"
    bits = getattr(args, "bits", None)
    bits_map = getattr(args, "bits_map", None)
    if bits is None and not bits_map:
        bits_map = getattr(config, "weight_bits", "") or ""
        if bits_map:
            print(f"[export] scheme from the checkpoint: {bits_map}")
        elif (run_meta or {}).get("stage") in ("qapt",) + HEAD_STAGES:
            raise ValueError(
                f"{os.path.basename(args.checkpoint)} is a "
                f"'{run_meta['stage']}' checkpoint with no config.weight_bits. "
                "Exporting it would silently ship plain W4 - numerics it was "
                "never post-trained for. Migrate it "
                "(scripts/migrate_checkpoint_numerics.py) or pass --bits-map.")
        else:
            bits = 4
    elif bits is None:
        bits = 4
    from .architecture import effective_kv_window
    kv_window = effective_kv_window(config)
    info = write_export(params, config, out, bits=bits, bits_map=bits_map,
                        tokenizer=get_tokenizer(config.vocab_size),
                        kv_window=kv_window)
    mb = info["bytes"] / 1e6
    w = f"mixed[{bits_map}]" if bits_map else f"W{bits}"
    print(f"wrote {info['path']}: {info['tensors']} tensors (+tokenizer), "
          f"{mb:.2f} MB ({w}A8)")
