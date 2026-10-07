"""query_data when the query names a column that is not there (bug #405).

An agent looking for the fc_* feature views (list_data shows only the data files, each under the
key "view") sent `SELECT view FROM information_schema.tables WHERE table_name LIKE 'fc_%'` three
times and got DuckDB's one candidate, "is_insertable_into". Listing the views now runs on
table_name with a note, and any other missing column is answered with the table's real columns.
"""

from __future__ import annotations

import duckdb
import pytest

from app import datasource as D
from app import objectives as O


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    con = duckdb.connect(":memory:")
    con.execute(f"COPY (SELECT TIMESTAMP '2024-01-02 10:00' AS SlotUtc, 1.5 AS Close, -2.0 AS GEX) "
                f"TO '{(d / 'gexbar10s.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT TIMESTAMP '2024-01-02 10:00' AS t, 0.1 AS fc_median) "
                f"TO '{(d / 'fc_close_5.parquet').as_posix()}' (FORMAT parquet)")
    con.close()
    return str(d)


def test_listing_the_views_by_a_view_column_runs_on_table_name(data_dir):
    got = D.query(data_dir, "SELECT view FROM information_schema.tables WHERE table_name LIKE 'fc_%' LIMIT 20")
    assert [r[0] for r in got["rows"]] == ["fc_close_5"]
    assert "table_name" in got["note"] and '"view"' in got["note"]
    got = D.query(data_dir, "SELECT DISTINCT name FROM information_schema.tables ORDER BY 1")
    assert [r[0] for r in got["rows"]] == ["fc_close_5", "gexbar10s"] and got["note"]
    # A query that works carries no note.
    assert "note" not in D.query(data_dir, "SELECT table_name FROM information_schema.tables")


def test_a_missing_column_is_answered_with_the_tables_columns(data_dir):
    with pytest.raises(D.DataError) as exc:
        D.query(data_dir, "SELECT Clos, GEX FROM gexbar10s")
    msg = str(exc.value)
    assert msg.startswith("Binder Error") and "Columns of gexbar10s: SlotUtc, Close, GEX." in msg
    assert 'Did you mean "Close"?' in msg and "\n" not in msg
    # A CTE of the same name is not described as the view.
    with pytest.raises(D.DataError) as exc:
        D.query(data_dir, "WITH b AS (SELECT Close FROM gexbar10s) SELECT GEX FROM b")
    assert "Columns of gexbar10s" in str(exc.value)


def test_another_miss_on_information_schema_lists_its_columns_and_the_tables(data_dir):
    with pytest.raises(D.DataError) as exc:
        D.query(data_dir, "SELECT foo FROM information_schema.tables")
    msg = str(exc.value)
    assert "Columns of information_schema.tables:" in msg and "table_name" in msg
    assert "Tables you can query: fc_close_5, gexbar10s." in msg


def test_objective_query_sees_the_feature_views_and_gets_the_same_help(data_dir, tmp_path, monkeypatch):
    """The objective endpoint (what the swarm calls): its fc_* views come from the feature catalog."""
    feats = tmp_path / "features"
    feats.mkdir()
    con = duckdb.connect(":memory:")
    con.execute(f"COPY (SELECT TIMESTAMP '2024-01-02 10:00' AS t, 0.2 AS fc_median) "
                f"TO '{(feats / 'f3.parquet').as_posix()}' (FORMAT parquet)")
    con.close()
    monkeypatch.setattr(O, "features_dir", lambda obj, cut: str(feats))
    monkeypatch.setattr(O, "_feature_catalog", lambda oid: [{"view": "fc_imb_oinet_d0_forecast_3", "path": "f3.parquet"}])
    obj = {"id": "o1"}
    got = O._insample_query(obj, data_dir, "SELECT DISTINCT view FROM information_schema.tables "
                                          "WHERE table_name LIKE 'fc_%' ORDER BY 1", 200)
    assert [r[0] for r in got["rows"]] == ["fc_close_5", "fc_imb_oinet_d0_forecast_3"] and got["note"]
    with pytest.raises(D.DataError) as exc:
        O._insample_query(obj, data_dir, "SELECT fc_medan FROM fc_imb_oinet_d0_forecast_3", 200)
    assert 'Did you mean "fc_median"? Columns of fc_imb_oinet_d0_forecast_3: t, fc_median.' in str(exc.value)


def test_list_data_names_the_objectives_forecast_views():
    """#405: in objective mode list_data named only the data files, so agents hunted for the fc_* views with
    `SELECT view FROM information_schema.tables`. They are listed now, by the column they forecast."""
    import swarm_runner as sr

    feats = [{"view": f"fc_gex_h{h}", "params": {"series": ["GEX"], "horizon": h}, "created_at": h} for h in (3, 6)]
    feats.append({"view": "fc_old", "recipe": {"request": {"column": "Close"}}, "created_at": 0})
    out = sr._with_forecast_views([{"view": "bars"}], feats)
    assert out["files"] == [{"view": "bars"}]
    assert out["forecasts"] == {"GEX": ["fc_gex_h6 (h6)", "fc_gex_h3 (h3)"], "Close": ["fc_old"]}
    assert "3 forecast views" in out["forecasts_note"] and "information_schema" not in out["forecasts_note"]
    assert sr._with_forecast_views({"files": []}, []) == {"files": []}
