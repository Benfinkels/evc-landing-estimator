"""Tests for CRM, UTM, GA4, wide/pivot, XLSX, and Google Ads EVC adapters."""

from pathlib import Path

import pandas as pd

from evc_landing_estimator.adapters import load_conversions, load_evc
from evc_landing_estimator.config import ColumnOverrides, Config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_ga4_comments_and_thousands():
    loaded = load_conversions(EXAMPLES / "ga4_daily_export.csv")
    assert loaded.shape == "aggregated_long"
    assert len(loaded.df) > 100
    assert loaded.df["conversions"].sum() > 50_000
    assert "Direct" in set(loaded.df["channel"])


def test_google_ads_title_rows_total_and_dashes(tmp_path):
    p = tmp_path / "ads.csv"
    p.write_text(
        "My YouTube Report\n"
        "Jan 1 - Jan 5, 2026\n"
        "Day,Engaged-view conversions\n"
        "2026-01-01,\"1,025\"\n"
        "2026-01-02,--\n"
        "2026-01-03,42\n"
        "Total,\"1,067\"\n",
        encoding="utf-8",
    )
    evc = load_evc(p)
    assert len(evc) == 3
    assert evc.loc[pd.Timestamp("2026-01-01")] == 1025.0
    assert evc.loc[pd.Timestamp("2026-01-02")] == 0.0
    assert evc.loc[pd.Timestamp("2026-01-03")] == 42.0


def test_crm_dedup_and_timezone_date_boundary(tmp_path):
    # America/New_York in January is UTC-5:
    # 2026-01-02T01:30:00Z -> 2026-01-01 20:30 EST (shifts date back to Jan 1)
    # 2026-01-02T04:59:00Z -> 2026-01-01 23:59 EST (shifts date back to Jan 1)
    # 2026-01-02T05:01:00Z -> 2026-01-02 00:01 EST (stays Jan 2)
    p = tmp_path / "crm.csv"
    p.write_text(
        "lead_id,created_at,utm_source,utm_medium\n"
        "L1,2026-01-02T01:30:00Z,(direct),(none)\n"
        "L2,2026-01-02T04:59:00Z,(direct),(none)\n"
        "L2,2026-01-02T04:59:00Z,(direct),(none)\n"
        "L3,2026-01-02T05:01:00Z,(direct),(none)\n",
        encoding="utf-8",
    )
    cfg = Config(source_tz="UTC", ads_tz="America/New_York")
    loaded = load_conversions(p, cfg)
    assert loaded.shape == "row_level"
    assert loaded.duplicates_removed == 1
    by_day = loaded.df.groupby("date")["conversions"].sum()
    assert by_day.loc[pd.Timestamp("2026-01-01")] == 2.0
    assert by_day.loc[pd.Timestamp("2026-01-02")] == 1.0


def test_wide_and_pivot_formats(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text(
        "Date,Direct,Affiliate\n"
        "2026-01-01,10,5\n"
        "2026-01-02,12,8\n",
        encoding="utf-8",
    )
    w_res = load_conversions(wide)
    assert w_res.shape == "aggregated_wide"
    assert w_res.df["conversions"].sum() == 35.0

    piv = tmp_path / "pivot.csv"
    piv.write_text(
        "Channel,2026-01-01,2026-01-02,2026-01-03\n"
        "Direct,10,12,14\n"
        "Affiliate,5,8,6\n",
        encoding="utf-8",
    )
    p_res = load_conversions(piv)
    assert p_res.shape == "aggregated_pivot"
    assert p_res.df["conversions"].sum() == 55.0


def test_xlsx_matches_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "Channel": ["Direct", "Affiliates"],
            "Conversions": [15, 7],
        }
    )
    csv_p = tmp_path / "c.csv"
    xlsx_p = tmp_path / "c.xlsx"
    df.to_csv(csv_p, index=False)
    df.to_excel(xlsx_p, index=False)

    r_csv = load_conversions(csv_p).df
    r_xlsx = load_conversions(xlsx_p).df
    pd.testing.assert_frame_equal(r_csv, r_xlsx)


def test_column_overrides_win(tmp_path):
    p = tmp_path / "custom.csv"
    p.write_text(
        "my_dt,my_src,my_cnt,ignore_date\n"
        "2026-01-01,rakuten,9,2020-01-01\n",
        encoding="utf-8",
    )
    cfg = Config(
        columns=ColumnOverrides(timestamp="my_dt", source="my_src", conversions="my_cnt")
    )
    res = load_conversions(p, cfg)
    assert res.df.loc[0, "date"] == pd.Timestamp("2026-01-01")
    assert res.df.loc[0, "source"] == "rakuten"
    assert res.df.loc[0, "conversions"] == 9.0


# --- European decimal commas & localized dates (issue #1) -------------------
from evc_landing_estimator.adapters import (  # noqa: E402
    _clean_numeric_series,
    _parse_localized_dates,
)


def test_clean_numeric_us_thousands_not_misread():
    # US exports: comma is a thousands group, never a decimal point.
    s = _clean_numeric_series(pd.Series(["1,025", "1,067", "12,500", "42"]))
    assert s.tolist() == [1025.0, 1067.0, 12500.0, 42.0]


def test_clean_numeric_eu_decimal_comma():
    # EU exports: comma is the decimal separator.
    s = _clean_numeric_series(pd.Series(["12,5", "0,25", "1,2345", "3,0"]))
    assert s.tolist() == [12.5, 0.25, 1.2345, 3.0]


def test_clean_numeric_eu_dot_grouping_and_mixed():
    # 1.234 -> one thousand two hundred thirty-four (dot = thousands group)
    # 1.234,56 -> 1234.56 (dot groups, comma decimals)
    s = _clean_numeric_series(pd.Series(["1.234", "1.234,56", "12.345.678"]))
    assert s.tolist() == [1234.0, 1234.56, 12345678.0]


def test_clean_numeric_unparseable_stays_nan():
    # Unparseable values must NOT be silently coerced to 0.
    s = _clean_numeric_series(pd.Series(["n/a", "abc", "42"]))
    assert pd.isna(s.iloc[0])
    assert pd.isna(s.iloc[1])
    assert s.iloc[2] == 42.0


def test_localized_dates_day_first_and_unambiguous_month_first():
    parsed = _parse_localized_dates(
        pd.Series(["03/04/2026", "13/04/2026", "04/13/2026", "2026-04-03"])
    )
    # 03/04 -> 3 April (day-first European default)
    assert parsed.iloc[0] == pd.Timestamp("2026-04-03")
    # 13/04 -> day 13 forces day-first
    assert parsed.iloc[1] == pd.Timestamp("2026-04-13")
    # 04/13 -> month 13 impossible, so it is month-first (US)
    assert parsed.iloc[2] == pd.Timestamp("2026-04-13")
    # ISO passes through
    assert parsed.iloc[3] == pd.Timestamp("2026-04-03")


def test_eu_google_ads_csv_end_to_end(tmp_path):
    # A Google Ads export with EU number formatting and DD/MM/YYYY dates.
    p = tmp_path / "ads_eu.csv"
    p.write_text(
        "Day,Engaged-view conversions\n"
        "03/04/2026,\"1.234,5\"\n"
        "04/04/2026,\"12,5\"\n"
        "05/04/2026,42\n",
        encoding="utf-8",
    )
    evc = load_evc(p)
    assert len(evc) == 3
    assert evc.loc[pd.Timestamp("2026-04-03")] == 1234.5
    assert evc.loc[pd.Timestamp("2026-04-04")] == 12.5
    assert evc.loc[pd.Timestamp("2026-04-05")] == 42.0
