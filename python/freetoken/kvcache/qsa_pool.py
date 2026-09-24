"""Paged GQA K/V + a per-token side slab of raw indexer keys (Qwen4-Exp QSA).

Qwen Sparse Attention scores mean-pooled blocks of RAW (pre-norm, pre-rope) indexer keys, so
every attention layer needs one ``index_head_dim``-wide bf16 key per cached token in addition
to its K/V. The attention type stays FULL (the dense path is exact for short contexts and the
model does the sparse selection itself), so this is the MHA pool -- same subset-of-layers
storage (``layer_ids``) -- plus the index slab, addressed by the same physical token rows as
the K/V slabs (``out_loc`` / page-table entries). Both slabs resize together in ``rebuild``
and the KV cost model budgets the slab (``kv_cost``) so the startup solve never over-commits.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .mha_pool import MHAKVCache


class QSAKVCache(MHAKVCache):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._index_head_dim = index_head_dim
        self._index_dtype = dtype
        self._alloc_index_slab()

    def _alloc_index_slab(self) -> None:
        num_storage_layers = self._kv_buffer.shape[1]
        rows = self._storage_shape[0]
        # Zero-init: unwritten rows (dummy page, freed pages) must stay finite if ever read.
        self._index_k_buffer = torch.zeros(
            (num_storage_layers, rows, self._index_head_dim),
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._index_k_buffer = None
        super().rebuild(num_pages)
        try:
            self._alloc_index_slab()
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        cache_per_page, fixed, page_tokens, reserve = super().kv_cost(config)
        mc = config.model_config
        num_kv_layers = sum(
            spec.num_layers for spec in mc.kv_cache_group_specs() if not spec.is_swa
        )
        index_bytes = int(mc.qsa_index_head_dim) * num_kv_layers * config.dtype.itemsize
        return cache_per_page + index_bytes * config.page_size, fixed, page_tokens, reserve

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        return kv + int(idx.numel() * idx.element_size()) // int(idx.shape[1]), swa

    def index_k_rows(self, layer_id: int) -> torch.Tensor:
        """Row-flat raw index keys for a paged-KV layer: ``[rows, index_head_dim]``."""
        return self._index_k_buffer[self._dense(layer_id)]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        self._index_k_buffer[self._dense(layer_id)][out_loc] = k.to(self._index_dtype)


__all__ = ["QSAKVCache"]
