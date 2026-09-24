"""Qwen4-Exp (Qwen3.8-Flash-Next) model-specific payload.

Carries everything the model module needs beyond the generic ``ModelConfig`` fields:
hyper-connection geometry, the PLE n-gram hashing constants (ported verbatim from HF
``modeling_qwen4_exp``: ``_splitmix64`` / ``_build_layer_multipliers`` / the per-head prime
vocab sizes) and the QSA (Qwen Sparse Attention) indexer geometry. Opaque to the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int,
                            seed: int) -> list[int]:
    """HF ``_build_layer_multipliers`` (returns python ints; odd, < 2**63 / vocab)."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def primes_after(start: int, count: int) -> list[int]:
    """The first ``count`` primes strictly greater than ``start``. Element ``i`` equals HF's
    ``_find_nth_prime_after(start, i + 1)`` (computed incrementally instead of rescanning)."""
    out, p = [], start
    while len(out) < count:
        p += 1
        while not _is_prime(p):
            p += 1
        out.append(p)
    return out


@dataclass(frozen=True)
class PLELayerSpec:
    """One PLE (Per-Layer Embedding) module: decoder layer ``layer_id`` (0-indexed) is the
    ``ple_index``-th entry of ``ple_layer_ids``."""

    layer_id: int
    ple_index: int
    multipliers: tuple[int, ...]  # [ngram_size] int64 hash multipliers
    head_vocab_sizes: tuple[int, ...]  # [ngram_heads] distinct primes
    head_offsets: tuple[int, ...]  # [ngram_heads] row offset of each head's table slice
    total_vocab_size: int
    padded_vocab_size: int


@dataclass(frozen=True)
class Qwen4ExpArgs:
    hidden_size: int
    vocab_size: int
    # hyper-connections
    hc_count: int
    hc_lowrank: int
    # QSA
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    max_position: int
    indexer_n_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    qsa_layer_ids: tuple[int, ...]
    # GDN output gate activation (Qwen4-Exp: sigmoid; Qwen3.5: silu)
    output_gate_type: str
    # PLE
    ple_layers: tuple[PLELayerSpec, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    # checkpoint location (the n-gram table is served from host memory, loaded out-of-band)
    model_path: str | None = None

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def dense_kv_limit(self) -> int:
        """Largest kv length for which QSA selects EVERY visible token (== dense causal
        attention): the number of complete blocks never exceeds the block budget."""
        return (self.block_topk + 1) * self.indexer_compress_ratio - 1

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads

    def qsa_slot(self, layer_id: int) -> int:
        return self.qsa_layer_ids.index(layer_id)

    def ple_for_layer(self, layer_id: int) -> PLELayerSpec | None:
        for spec in self.ple_layers:
            if spec.layer_id == layer_id:
                return spec
        return None


def build_ple_layer_spec(*, layer_id: int, ple_index: int, vocab_size: int, ngram_size: int,
                         heads_per_ngram: int, ngram_vocab_size_base: int,
                         make_divisible_by: int, seed: int) -> PLELayerSpec:
    """Mirror of HF ``Qwen4ExpTextNGramEmbedding.__init__``: head ``h`` of PLE module ``i``
    uses the ``(i * ngram_heads + h + 1)``-th prime after ``ngram_vocab_size_base - 1``."""
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    primes = primes_after(ngram_vocab_size_base - 1, (ple_index + 1) * ngram_heads)
    sizes = primes[ple_index * ngram_heads:(ple_index + 1) * ngram_heads]
    offsets, total = [], 0
    for s in sizes:
        offsets.append(total)
        total += s
    padded = math.ceil(total / make_divisible_by) * make_divisible_by
    return PLELayerSpec(
        layer_id=layer_id,
        ple_index=ple_index,
        multipliers=tuple(build_layer_multipliers(vocab_size, ngram_size, ple_index, seed)),
        head_vocab_sizes=tuple(sizes),
        head_offsets=tuple(offsets),
        total_vocab_size=total,
        padded_vocab_size=padded,
    )


__all__ = [
    "Qwen4ExpArgs",
    "PLELayerSpec",
    "build_ple_layer_spec",
    "build_layer_multipliers",
    "primes_after",
    "splitmix64",
]
