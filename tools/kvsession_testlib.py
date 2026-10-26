"""GPU-test helpers for the KV session store: load libnomos_kernel-muse.so + Glimmer once per process and drive it
through the same FFI pyhost uses. Nothing else may hold the GPU (bash pyhost/serve_nomos.sh down)."""
import ctypes
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SO = os.environ.get("SO_PATH", str(ROOT / "libnomos_kernel-muse.so"))
WEIGHTS = os.environ.get("WEIGHTS", os.path.expanduser("~/nomos_data/muse-glimmer-flat-nvfp4/"))
TOK_DIR = os.path.expanduser("~/nomos_data/muse-glimmer-tokenizer")
DFLASH_DIR = os.path.expanduser("~/nomos_data/muse-assistant-flat-q4")
VOCAB = 202048                 # lib/model_config.mojo VOCAB for the Muse profile
BOS = 200000                   # <|begin_of_text|>
STOP = (200001, 200008)        # <|end_of_text|>, <|eot|>: serve_protocol's stop set
# The laptop's production engine env: pyhost/serve_nomos.sh (x86_64 branch) + ~/.nomos_host_env.
# KVSESSION_KV=int4 runs the same tests under INT4 block-32 with the int8-attention path (NOMOS_Q4_DP4A=1), the
# other KV mode the store supports. That path widens the nibbles through the cached fp32 q8-scale tail, so the
# tail is exercised. Without the flag INT4 attends through the bf16 dequant, which never reads the tail.
KV_MODE = os.environ.get("KVSESSION_KV", "int8")
assert KV_MODE in ("int8", "int4"), f"KVSESSION_KV={KV_MODE!r}: use int8 or int4"
EXPECTED_BPT = {"int8": 27456, "int4": 15808}[KV_MODE]   # Glimmer KV bytes per token (spec §2)
BASE_ENV = {
    "NOMOS_WEIGHT_NVFP4": "1", "NOMOS_W4A4": "1", "NOMOS_PRECISION_BITS": "16",
    "NOMOS_KV_QUANT": "1", "NOMOS_KV_INT4": "1" if KV_MODE == "int4" else "0", "NOMOS_KV_I4_BLOCK": "32",
    "NOMOS_KV_SWA": "0", "NOMOS_MAX_SEQ": "8192", "NOMOS_BATCHED_PREFILL": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NOMOS_VERIFY_BLOCK_ATTN": "0",
}
if KV_MODE == "int4":
    BASE_ENV["NOMOS_Q4_DP4A"] = "1"   # int8 attention over the INT4 KV (engine_decode/engine_prefill use_int8_attn)
I32, I64 = ctypes.c_int32, ctypes.c_int64
SIGNATURES = {
    "nomos_init": ([I64], I64),
    "nomos_shutdown": ([I64], I32),
    "nomos_prefill": ([I64, I64, I32, I64], I32),
    "nomos_prefill_cont": ([I64, I64, I32, I64], I32),
    "nomos_decode_step": ([I64, I32, I64], I32),
    "nomos_decode_step_token": ([I64, I32, I64], I32),
    "nomos_kv_cache_len": ([I64], I32),
    "nomos_kv_set_len": ([I64, I32], I32),
    "nomos_reset_kv": ([I64], I32),
    "nomos_generate": ([I64, I64, I32, I32, I64, I32, ctypes.c_float, ctypes.c_float, ctypes.c_float], I32),
    "nomos_live_tokens": ([I64, I64, I32], I32),
    "nomos_kv_bytes_per_token": ([I64], I64),
    "nomos_kv_export": ([I64, I32, I32, I64], I32),
    "nomos_kv_import": ([I64, I32, I32, I64, I32, I64], I32),
    "nomos_kv_truncate": ([I64, I32], I32),
    "nomos_session_status": ([I64], I32),
    "nomos_session_open": ([I64, I64, I32], I32),
    "nomos_session_prepare": ([I64, I64, I32, I32], I32),
    "nomos_session_commit": ([I64], I32),
    "nomos_session_stats": ([I64, I64], I32),
    "nomos_dflash_load": ([I64, I64], I32),
    "nomos_dflash_cache_len": ([I64], I32),
    "nomos_dflash_set_len": ([I64, I32], I32),
    "nomos_dflash_reset": ([I64], I32),
    "nomos_dflash_draft_block": ([I64, I32, I32, I64], I32),
    "nomos_dflash_verify_fused": ([I64, I64, I32, I32, I64], I32),
    "nomos_dflash_append_verify_context": ([I64, I32, I32], I32),
}


def _arr(ids):
    a = np.ascontiguousarray(ids, dtype=np.int32)
    return a, a.ctypes.data


class Kernel:
    def __init__(self, extra_env=None):
        for k, v in {**BASE_ENV, **(extra_env or {})}.items():
            os.environ[k] = str(v)
        self.max_seq = int(os.environ["NOMOS_MAX_SEQ"])
        self.lib = ctypes.CDLL(SO)
        for name, (args, res) in SIGNATURES.items():
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.argtypes, fn.restype = args, res
        self._wdir = ctypes.create_string_buffer(WEIGHTS.encode())
        self.h = self.lib.nomos_init(ctypes.addressof(self._wdir))
        if not self.h:
            raise RuntimeError("nomos_init failed")
        self.logits = np.zeros(VOCAB, dtype=np.float32)

    def shutdown(self):
        self.lib.nomos_shutdown(self.h)

    # -- the logits engine --------------------------------------------------------
    def prefill(self, ids):
        a, p = _arr(ids)
        rc = self.lib.nomos_prefill(self.h, p, len(a), self.logits.ctypes.data)
        assert rc == 0, f"nomos_prefill rc={rc}"
        return self.logits.copy()

    def prefill_cont(self, ids):
        a, p = _arr(ids)
        rc = self.lib.nomos_prefill_cont(self.h, p, len(a), self.logits.ctypes.data)
        assert rc == 0, f"nomos_prefill_cont rc={rc}"
        return self.logits.copy()

    def step(self, tok):
        rc = self.lib.nomos_decode_step(self.h, int(tok), self.logits.ctypes.data)
        assert rc == 0, f"nomos_decode_step rc={rc}"
        return self.logits.copy()

    def step_token(self, tok):
        """The greedy fast path (nomos_decode_step_token): appends tok, returns the argmax token."""
        out = np.zeros(1, dtype=np.int32)
        rc = self.lib.nomos_decode_step_token(self.h, int(tok), out.ctypes.data)
        assert rc == 0, f"nomos_decode_step_token rc={rc}"
        return int(out[0])

    def greedy(self, logits, n):
        """Greedy decode like host._GenMixin.generate: every non-stop token is stepped, so it has KV; a stop
        token ends the loop without being stepped."""
        out, lg = [], logits
        for _ in range(n):
            t = int(np.argmax(lg))
            out.append(t)
            if t in STOP:
                break
            lg = self.step(t)
        return out

    def cache_len(self):
        return self.lib.nomos_kv_cache_len(self.h)

    def set_len(self, n):
        return self.lib.nomos_kv_set_len(self.h, n)

    def reset(self):
        assert self.lib.nomos_reset_kv(self.h) == 0

    def live_tokens(self):
        out = np.zeros(self.max_seq, dtype=np.int32)       # the engine never holds more than NOMOS_MAX_SEQ tokens
        n = self.lib.nomos_live_tokens(self.h, out.ctypes.data, len(out))
        assert 0 <= n <= len(out), f"nomos_live_tokens returned {n} for a {len(out)}-token buffer"
        return out[:n].tolist()

    # -- session primitives (Task 5) -------------------------------------------------
    def bytes_per_token(self):
        return self.lib.nomos_kv_bytes_per_token(self.h)

    def export(self, a, b):
        buf = np.zeros((b - a) * self.bytes_per_token(), dtype=np.uint8)
        rc = self.lib.nomos_kv_export(self.h, a, b, buf.ctypes.data)
        assert rc == 0, f"nomos_kv_export rc={rc}"
        return buf

    def import_(self, a, b, buf, src_tokens, tokens):
        t, tp = _arr(tokens)
        return self.lib.nomos_kv_import(self.h, a, b, buf.ctypes.data, src_tokens, tp)

    def truncate(self, p):
        return self.lib.nomos_kv_truncate(self.h, p)

    # -- the store (Task 9) -------------------------------------------------------------
    def status(self):
        return self.lib.nomos_session_status(self.h)

    def session_open(self):
        """nomos_session_open (ruling R79): (status, the store's directory as the kernel names it; empty unless open)."""
        buf = ctypes.create_string_buffer(4096)
        st = self.lib.nomos_session_open(self.h, ctypes.addressof(buf), len(buf))
        return st, buf.value.decode()

    def prepare(self, ids, spec=False):
        a, p = _arr(ids)
        return self.lib.nomos_session_prepare(self.h, p, len(a), 1 if spec else 0)

    def commit(self):
        return self.lib.nomos_session_commit(self.h)

    def stats(self):
        out = np.zeros(10, dtype=np.int64)
        assert self.lib.nomos_session_stats(self.h, out.ctypes.data) == 0
        return dict(zip(["prepares", "prompt_tokens", "reused_tokens", "restores_ram", "restores_nvme", "forks",
                         "ram_bytes", "nvme_bytes", "ram_evictions", "nvme_evictions"], out.tolist()))

    def turn(self, ids, n_gen, spec=False):
        """One chat turn through the store: prepare, prefill what is not cached, generate, commit."""
        c = self.prepare(ids, spec=spec)
        assert c >= 0, f"prepare rc={c}"
        assert self.cache_len() == c and self.live_tokens() == list(ids[:c]), "engine must hold exactly ids[:c]"
        lg = self.prefill(ids) if c == 0 else self.prefill_cont(ids[c:])
        gen = self.spec_trace(lg, n_gen)[0] if spec else self.greedy(lg, n_gen)
        assert self.commit() >= 0
        return c, gen

    # -- DFlash (Tasks 11-13) -----------------------------------------------------------------
    def dflash_load(self, path=DFLASH_DIR):
        if not path.endswith("/"):
            path += "/"          # nomos_dflash_load concatenates dir + filename: the loader requires a trailing slash
        self._ddir = ctypes.create_string_buffer(path.encode())
        rc = self.lib.nomos_dflash_load(self.h, ctypes.addressof(self._ddir))
        assert rc == 0, f"nomos_dflash_load rc={rc}"

    def dflash_len(self):
        return self.lib.nomos_dflash_cache_len(self.h)

    def spec_trace(self, logits, max_new=64, vb=8):
        """Greedy DFlash from the engine's current state: host.spec_generate_stream's draft/verify/accept cycles, giving
        the same tokens. Not its end state: it makes no store calls, and when its budget ends at the exact M=1 step it keeps
        that step's KV row, which the host drops since ecc2140 (ledger R54); nor does it settle a turn (R55).
        Returns (tokens, accepted drafts per cycle)."""
        k_max, out15 = vb - 1, np.zeros(15, dtype=np.int32)
        vlog = np.zeros((vb, VOCAB), dtype=np.float32)
        seed = int(np.argmax(logits))
        toks, acc = [seed], []
        if seed in STOP:
            return toks, acc
        L = self.cache_len()
        while len(toks) < max_new:
            r = max_new - len(toks)
            k = min(k_max, r - 1) if r > 1 else 0
            if k <= 0:
                toks.append(int(np.argmax(self.step(seed))))
                break
            assert self.lib.nomos_dflash_draft_block(self.h, seed, L, out15.ctypes.data) == 15
            drafts = [int(x) for x in out15[:k]]
            rows = np.array([seed] + drafts, dtype=np.int32)
            assert self.lib.nomos_dflash_verify_fused(self.h, rows.ctypes.data, k + 1, L, vlog.ctypes.data) == 0
            tgt = [int(np.argmax(vlog[i])) for i in range(k + 1)]
            n = 0
            while n < k and drafts[n] == tgt[n]:
                n += 1
            acc.append(n)
            assert self.set_len(L + 1 + n) == 0
            assert self.lib.nomos_dflash_append_verify_context(self.h, 0, n + 1) > 0
            for t in drafts[:n] + [tgt[n]]:
                toks.append(t)
                if t in STOP or len(toks) >= max_new:
                    return toks, acc
            L, seed = L + 1 + n, tgt[n]
        return toks, acc


# -- where the store keeps its files (ledger R35) ------------------------------------------------------------------------
# The store's directory is <NOMOS_KVSTORE_DIR>/muse-<16 hex of fnv(BUILD_ID, numerics_id)>, where numerics_id mixes the
# weights directory's identity with the engine's precision flags. These re-implement lib/kvstore/key.mojo
# (weights_id_of, numerics_id_of) and lib/kv_session.mojo (session_store) on their own, so a test can predict the
# directory without loading the model. If the two disagree, the store lands in a directory the test does not expect.
_M64, _FNV0, _FNV_P = (1 << 64) - 1, 0xCBF29CE484222325, 0x100000001B3


def _fnv_u64(h, v):
    for i in range(8):
        h = ((h ^ ((v >> (8 * i)) & 0xFF)) * _FNV_P) & _M64
    return h


def weights_id_of(path):
    """The hash of convert_manifest.json plus every file's name and size, order-independent over the files."""
    h = _FNV0
    manifest = os.path.join(path, "convert_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "rb") as f:
            for b in f.read():
                h = ((h ^ b) * _FNV_P) & _M64
    files = 0
    for name in os.listdir(path):
        try:
            size = os.stat(os.path.join(path, name)).st_size
        except OSError:
            continue
        e = _FNV0
        for b in name.encode():
            e = ((e ^ b) * _FNV_P) & _M64
        files = (files + _fnv_u64(e, size)) & _M64
    return _fnv_u64(h, files)


def numerics_id(env, weights=WEIGHTS):
    """GemmaEngine.numerics_id of an engine started with this environment on this weights directory."""
    def on(name):
        return float(env.get(name, 0) or 0) > 0.5
    w4a4 = on("NOMOS_W4A4")
    h = _fnv_u64(_FNV0, weights_id_of(weights))
    h = _fnv_u64(h, int(float(env.get("NOMOS_PRECISION_BITS", 32))))
    h = _fnv_u64(h, 1 if (on("NOMOS_WEIGHT_NVFP4") or w4a4) else 0)    # W4A4 needs NVFP4 weights: it implies them
    h = _fnv_u64(h, 1 if w4a4 else 0)
    return _fnv_u64(h, 1 if on("NOMOS_GDN_NVFP4") else 0)


def store_dir_name(extra_env=None, weights=WEIGHTS):
    """The name of the store's directory under NOMOS_KVSTORE_DIR for a Kernel(extra_env). Loads no model."""
    lib = ctypes.CDLL(SO)
    lib.nomos_build_id.restype = I64
    env = {**BASE_ENV, **(extra_env or {})}
    return "muse-%016x" % _fnv_u64(lib.nomos_build_id() & _M64, numerics_id(env, weights))


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def doc_tokens(n):
    """BOS + the first n-1 tokens of docs/MEASUREMENT_TRAPS.md: deterministic real text, up to ~9K tokens."""
    ids = _tokenizer().encode((ROOT / "docs/MEASUREMENT_TRAPS.md").read_text(), add_special_tokens=False)
    assert len(ids) >= n - 1, f"doc has only {len(ids)} tokens"
    return [BOS] + ids[: n - 1]


def chat_tokens(messages):
    t = _tokenizer()
    return t.encode(t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)
