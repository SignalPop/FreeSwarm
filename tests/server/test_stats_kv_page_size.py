"""/v1/stats reports KV capacity in the pool's own page size.

DSV4 allocates pages of many tokens; reporting config.page_size (1) turned DeepSeek-V4's
1024 x 128-token pages into a "1K" window, and the swarm then squeezed every prompt into 1K.
"""

from types import SimpleNamespace

import freetoken.server.stats as stats


def _state(pools):
    tr = stats.StatsTracker()
    tr.kv_total_pages, tr.kv_used_pages = 1024, 10
    config = SimpleNamespace(page_size=1, model_config=None, served_model_name="m")
    return SimpleNamespace(stats=tr, config=config, ready_at=None, cache_pools=pools)


def test_kv_uses_allocated_pool_page_size(monkeypatch):
    monkeypatch.setattr(stats, "derive_model_card", lambda *a, **k: {})
    kv = stats.build_stats(_state({"page_size": 128, "num_pages": 1024}), 0, 0)["kv"]
    assert kv["page_size"] == 128
    assert kv["total_pages"] * kv["page_size"] == 131072


def test_kv_falls_back_to_config_page_size(monkeypatch):
    monkeypatch.setattr(stats, "derive_model_card", lambda *a, **k: {})
    kv = stats.build_stats(_state(None), 0, 0)["kv"]
    assert kv["page_size"] == 1 and kv["total_pages"] == 1024
