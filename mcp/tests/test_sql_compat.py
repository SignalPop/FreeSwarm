"""SQL other engines take that polars SQL does not, in task_query (bug #421)."""
import datetime as dt

import polars as pl
import pytest

from taskkit.task import safe_sql


@pytest.fixture
def ctx():
    return pl.SQLContext(rows=pl.DataFrame({"t": [dt.datetime(2026, 1, 5, 14, 31, 7), dt.datetime(2026, 1, 6, 15, 2)],
                                            "Close": [1.0, 2.0]}).lazy(), eager=False)


def test_time_functions_are_rewritten(ctx):
    out = safe_sql(ctx, "SELECT hour(t) AS h, MINUTE(t) m, dayofweek(t) d, date(t) AS day, year(t) y FROM rows "
                        "WHERE hour(t) >= 14 ORDER BY t").collect()
    assert out.row(0) == (14, 31, 1, dt.date(2026, 1, 5), 2026)
    nested = safe_sql(ctx, "SELECT hour(t) * 60 + minute(t) AS mins, avg(Close) FROM rows "
                           "GROUP BY hour(t) * 60 + minute(t) ORDER BY mins").collect()
    assert nested["mins"].to_list() == [871, 902]


def test_information_schema_lists_the_rows_columns(ctx):
    out = safe_sql(ctx, "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_name = 'rows' ORDER BY ordinal_position").collect()
    assert out["column_name"].to_list() == ["t", "Close"]
    assert safe_sql(ctx, "select table_name from INFORMATION_SCHEMA.TABLES").collect()["table_name"].to_list() == ["rows"]


def test_files_stay_refused(ctx):
    with pytest.raises(ValueError, match="reading files"):
        safe_sql(ctx, "SELECT * FROM read_parquet('x.parquet')")
