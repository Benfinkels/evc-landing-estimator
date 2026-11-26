"""Input adapters for CRM, UTM, GA4 conversions, Google Ads EVCs, and control files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from evc_landing_estimator.config import ColumnOverrides, Config

# Ordered synonym tables (case-insensitive substring match; exact match preferred first)
TIMESTAMP_SYNONYMS = ("converted_at", "timestamp", "created_at", "created", "submitted", "date", "day", "time")
ID_SYNONYMS = ("lead_id", "order_id", "transaction_id", "conversion_id", "id")
SOURCE_SYNONYMS = ("utm_source", "session source", "lead source", "lead_source", "source")
MEDIUM_SYNONYMS = ("utm_medium", "session medium", "medium")
CAMPAIGN_SYNONYMS = ("utm_campaign", "campaign")
TERM_SYNONYMS = ("utm_term", "keyword", "query", "term")
CHANNEL_SYNONYMS = ("default channel group", "channel grouping", "channel_group", "channel")
CONVERSIONS_SYNONYMS = ("key events", "conversions", "conversion_count", "count")
EVC_SYNONYMS = ("engaged-view conv", "engaged view conv", "evc")


@dataclass
class LoadedConversions:
    df: pd.DataFrame  # columns: date, source, medium, campaign, term, channel, conversions
    shape: str  # row_level | aggregated_long | aggregated_wide | aggregated_pivot
    duplicates_removed: int = 0
    detected_columns: dict[str, str | None] = field(default_factory=dict)


# Tokens that represent an explicit "no data" cell and are legitimately 0.
_NULL_TOKENS = frozenset({"--", "-", "", "nan", "none", "null", "n/a", "na"})


def _looks_european(raw: list[str]) -> bool:
    """True if a handful of numeric-looking strings follow EU conventions.

    The decision is deliberately conservative so that plain US exports
    (``1,025`` meaning one thousand and twenty-five) are never misread as
    European decimals.  A comma is only taken as the decimal separator when
    it is *unambiguously* a decimal point -- i.e. it is not followed by
    exactly three digits (which would be a thousands group), and it is not
    preceded by a dot (which would make the dot the decimal point).  A dot is
    only taken as a thousands separator when it groups in threes *and* there
    is no comma at all.
    """
    for token in raw:
        s = token.strip()
        if not s or s.lower() in _NULL_TOKENS:
            continue
        # comma decimal separator, unambiguously EU:
        #   - decimal part is not exactly 3 digits (1,5 / 1,25 / 1,2345)
        #   - or a dot grouping precedes it (1.234,5)
        if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})*,\d+", s):
            frac = s.rsplit(",", 1)[1]
            if len(frac) != 3 or "." in s:
                return True
            # ambiguous: 1,234 could be US thousands or EU decimal -> skip
            continue
        # dot thousands separator without any comma: 1.234 / 12.345.678
        if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", s):
            return True
    return False


def _parse_numeric_token(token: str, european: bool) -> float:
    """Parse a single numeric cell into a float, honouring the locale."""
    if token is None:
        return np.nan
    s = str(token).strip()
    if s.lower() in _NULL_TOKENS:
        return 0.0 if s in ("--", "-") else np.nan
    s = s.replace("\u00a0", "").replace("\u202f", "")  # NBSP / narrow NBSP
    s = s.replace("\u2212", "-")  # unicode minus
    s = re.sub(r"^[^\d+\-.,]+|[^\d]+$", "", s)  # leading/trailing currency & units
    if not s:
        return np.nan
    if european:
        # EU: '.' groups thousands, ',' is the decimal point
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            frac = s.rsplit(",", 1)[1]
            # only treat as decimal when unambiguous; otherwise thousands group
            if len(frac) == 3 and s.count(",") == 1 and re.fullmatch(r"[+-]?\d{1,3},\d{3}", s):
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
        elif re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
    else:
        # US: ',' groups thousands, '.' is the decimal point
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a column to float, auto-detecting US vs European conventions.

    A column is treated as European when several of its numeric-looking cells
    unambiguously follow EU conventions (comma decimal / dot grouping).  Cells
    that cannot be parsed become NaN; callers decide whether that is an error.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    raw = [str(v) for v in s.tolist() if v is not None]
    european = _looks_european(raw)
    return pd.Series(
        [_parse_numeric_token(v, european) for v in s.tolist()],
        index=s.index,
        dtype="float64",
    )


def _find_header_row(rows: list[list[str]], extra_tokens: tuple[str, ...] = ()) -> int:
    tokens = ("date", "day", "timestamp", "created", "submitted", "converted") + extra_tokens
    for idx, row in enumerate(rows[:15]):
        non_empty = [str(x).strip() for x in row if str(x).strip() and str(x).strip().lower() != "nan"]
        if len(non_empty) < 2:
            continue
        joined = " ".join(x.lower() for x in non_empty)
        if any(tok in joined for tok in tokens):
            return idx
        # Also handle pivot tables where >50% of headers[1:] parse as dates
        if len(non_empty) >= 3:
            sample = non_empty[1:10]
            parsed = pd.to_datetime(pd.Series(sample), errors="coerce", format="mixed").notna().sum()
            if parsed > len(sample) * 0.5:
                return idx
    return 0


def _read_table(path: str | Path, extra_header_tokens: tuple[str, ...] = ()) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        preview = pd.read_excel(p, header=None, nrows=15, dtype=str)
        rows = preview.fillna("").values.tolist()
        hdr = _find_header_row(rows, extra_header_tokens)
        df = pd.read_excel(p, header=hdr)
    else:
        import csv

        # Strip leading comment/blank lines, then scan with csv.reader so 1-column title lines don't choke pandas
        raw_lines = p.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        skip_comments = 0
        while skip_comments < len(raw_lines) and (
            raw_lines[skip_comments].lstrip().startswith("#") or not raw_lines[skip_comments].strip()
        ):
            skip_comments += 1
        preview_rows = list(csv.reader(raw_lines[skip_comments : skip_comments + 15]))
        hdr = _find_header_row(preview_rows, extra_header_tokens)
        df = pd.read_csv(p, header=0, skiprows=skip_comments + hdr, dtype=str)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _match_column(
    cols: list[str],
    synonyms: tuple[str, ...],
    override: str | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    if override is not None:
        if override not in cols:
            raise ValueError(f"Requested column override {override!r} not found in columns: {cols}")
        return override
    blocked = exclude or set()
    available = [c for c in cols if c not in blocked]
    lower_map = {c: c.lower().strip() for c in available}

    # Exact match pass first
    for syn in synonyms:
        for col, low in lower_map.items():
            if low == syn:
                return col
    # Substring pass (avoid matching 'id' inside 'paid' or 'valid')
    for syn in synonyms:
        for col, low in lower_map.items():
            if syn == "id":
                if low == "id" or low.endswith("_id") or low.endswith(" id"):
                    return col
            elif syn in low:
                return col
    return None


def _is_pivot_table(df: pd.DataFrame) -> bool:
    if len(df.columns) < 4:
        return False
    sample = list(df.columns[1:12])
    parsed = pd.to_datetime(pd.Series(sample), errors="coerce", format="mixed").notna().sum()
    return parsed > len(sample) * 0.5


def _has_time_of_day(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    return bool(sample.str.contains(r"[T\s]\d{1,2}:\d{2}", regex=True).any())


_DMY_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?:[ T].*)?$")
_YMD_RE = re.compile(r"^\s*(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?:[ T].*)?$")


def _parse_localized_dates(series: pd.Series) -> pd.Series:
    """Parse day-first (DD/MM/YYYY) and ISO (YYYY-MM-DD) strings unambiguously.

    ``pandas.to_datetime`` reads ``03/04/2026`` month-first (Mar 4), which is
    wrong for European exports (3 April).  Rows whose day/month layout is
    unambiguous are parsed explicitly; anything else is left to the caller.
    Returns a datetime64 Series with NaT where a value was not recognized.
    """
    out = []
    for value in series.astype(str):
        s = value.strip()
        m = _YMD_RE.match(s)
        if m:
            y, mo, d = (int(x) for x in m.groups())
            out.append(pd.Timestamp(year=y, month=mo, day=d))
            continue
        m = _DMY_RE.match(s)
        if m:
            a, b, y = (int(x) for x in m.groups())
            # a/b : if a > 12 it must be the day (day-first); if b > 12 it must be
            # the month (month-first, e.g. US 04/13/2026). Ambiguous -> day-first,
            # matching the European locale this adapter targets.
            if a > 12 and b <= 12:
                d, mo = a, b
            elif b > 12 and a <= 12:
                mo, d = a, b
            else:
                d, mo = a, b
            try:
                out.append(pd.Timestamp(year=y, month=mo, day=d))
            except ValueError:
                out.append(pd.NaT)
            continue
        out.append(pd.NaT)
    return pd.Series(out, index=series.index, dtype="datetime64[ns]")


def _normalize_dates(series: pd.Series, source_tz: str, ads_tz: str, is_row_level: bool) -> pd.Series:
    if not is_row_level:
        localized = _parse_localized_dates(series)
        if localized.notna().all():
            return localized.dt.normalize()
        dt = pd.to_datetime(series, errors="coerce", format="mixed")
        return dt.dt.normalize()

    src_zone = ZoneInfo(source_tz)
    ads_zone = ZoneInfo(ads_tz)

    # Try ISO/UTC-aware parse first; fall back to element-wise mixed parse if tz offsets vary
    try:
        dt = pd.to_datetime(series, errors="coerce", format="mixed", utc=False)
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(src_zone, ambiguous="NaT", nonexistent="shift_forward")
        dt_ads = dt.dt.tz_convert(ads_zone)
        return dt_ads.dt.tz_localize(None).dt.normalize()
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        dt_ads = parsed.dt.tz_convert(ads_zone)
        return dt_ads.dt.tz_localize(None).dt.normalize()


def load_conversions(source: str | Path | pd.DataFrame, config: Config | None = None) -> LoadedConversions:
    """Load CRM, UTM, or GA4 conversions (CSV/XLSX/DataFrame) into canonical pre-mapped form."""
    cfg = config or Config()
    ov: ColumnOverrides = cfg.columns
    df = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source)

    # 1. Pivot table (dates as column headers)
    if ov.timestamp is None and _is_pivot_table(df):
        id_col = df.columns[0]
        melted = df.melt(id_vars=[id_col], var_name="_date", value_name="_conv")
        dates = _normalize_dates(melted["_date"], cfg.source_tz, cfg.ads_tz, is_row_level=False)
        conv = _clean_numeric_series(melted["_conv"]).fillna(0.0)
        out = pd.DataFrame(
            {
                "date": dates,
                "source": "",
                "medium": "",
                "campaign": "",
                "term": "",
                "channel": melted[id_col].fillna("").astype(str).str.strip(),
                "conversions": conv,
            }
        ).dropna(subset=["date"])
        return LoadedConversions(df=out.reset_index(drop=True), shape="aggregated_pivot", detected_columns={"channel": id_col})

    cols = list(df.columns)
    used: set[str] = set()

    ts_col = _match_column(cols, TIMESTAMP_SYNONYMS, ov.timestamp)
    if ts_col is None:
        raise ValueError(f"Could not detect a date/timestamp column in {cols}. Pass --col-timestamp.")
    used.add(ts_col)

    id_col = _match_column(cols, ID_SYNONYMS, ov.id, exclude=used)
    if id_col:
        used.add(id_col)

    src_col = _match_column(cols, SOURCE_SYNONYMS, ov.source, exclude=used)
    if src_col:
        used.add(src_col)

    med_col = _match_column(cols, MEDIUM_SYNONYMS, ov.medium, exclude=used)
    if med_col:
        used.add(med_col)

    camp_col = _match_column(cols, CAMPAIGN_SYNONYMS, ov.campaign, exclude=used)
    if camp_col:
        used.add(camp_col)

    term_col = _match_column(cols, TERM_SYNONYMS, ov.term, exclude=used)
    if term_col:
        used.add(term_col)

    chan_col = _match_column(cols, CHANNEL_SYNONYMS, ov.channel, exclude=used)
    if chan_col:
        used.add(chan_col)

    conv_col = _match_column(cols, CONVERSIONS_SYNONYMS, ov.conversions, exclude=used)

    # 2. Wide format check: date column + >=2 numeric value columns and no channel/source/medium column
    if not any((src_col, med_col, chan_col)) and conv_col is None:
        other_cols = [c for c in cols if c not in used]
        numeric_cols = [c for c in other_cols if _clean_numeric_series(df[c]).notna().mean() > 0.7]
        if len(numeric_cols) >= 2:
            melted = df[[ts_col] + numeric_cols].melt(id_vars=[ts_col], var_name="_chan", value_name="_conv")
            dates = _normalize_dates(melted[ts_col], cfg.source_tz, cfg.ads_tz, is_row_level=False)
            conv = _clean_numeric_series(melted["_conv"]).fillna(0.0)
            out = pd.DataFrame(
                {
                    "date": dates,
                    "source": "",
                    "medium": "",
                    "campaign": "",
                    "term": "",
                    "channel": melted["_chan"].astype(str).str.strip(),
                    "conversions": conv,
                }
            ).dropna(subset=["date"])
            return LoadedConversions(
                df=out.reset_index(drop=True),
                shape="aggregated_wide",
                detected_columns={"timestamp": ts_col},
            )

    if not any((src_col, med_col, chan_col)):
        raise ValueError(
            f"At least one source, medium, or channel column is required. Found columns: {cols}"
        )

    # 3. Row-level vs aggregated long
    is_row_level = (id_col is not None) or (conv_col is None) or _has_time_of_day(df[ts_col])

    # Handle combined "source / medium" column (e.g. GA4 "Session source / medium")
    if src_col and med_col is None and "medium" in src_col.lower():
        parts = df[src_col].fillna("").astype(str).str.split("/", n=1, expand=True)
        src_vals = parts[0].str.strip()
        med_vals = parts[1].fillna("").str.strip() if parts.shape[1] > 1 else pd.Series("", index=df.index)
    else:
        src_vals = df[src_col].fillna("").astype(str).str.strip() if src_col else pd.Series("", index=df.index)
        med_vals = df[med_col].fillna("").astype(str).str.strip() if med_col else pd.Series("", index=df.index)

    camp_vals = df[camp_col].fillna("").astype(str).str.strip() if camp_col else pd.Series("", index=df.index)
    term_vals = df[term_col].fillna("").astype(str).str.strip() if term_col else pd.Series("", index=df.index)
    chan_vals = df[chan_col].fillna("").astype(str).str.strip() if chan_col else pd.Series("", index=df.index)

    dups_removed = 0
    work = pd.DataFrame(
        {
            "_raw_ts": df[ts_col],
            "source": src_vals,
            "medium": med_vals,
            "campaign": camp_vals,
            "term": term_vals,
            "channel": chan_vals,
            "conversions": _clean_numeric_series(df[conv_col]).fillna(0.0) if conv_col else 1.0,
        }
    )

    if id_col is not None:
        work["_id"] = df[id_col].astype(str).str.strip()
        before = len(work)
        work = work.drop_duplicates(subset=["_id"], keep="first").drop(columns=["_id"])
        dups_removed = before - len(work)

    work["date"] = _normalize_dates(work["_raw_ts"], cfg.source_tz, cfg.ads_tz, is_row_level=is_row_level)
    work = work.dropna(subset=["date"]).drop(columns=["_raw_ts"]).reset_index(drop=True)

    return LoadedConversions(
        df=work[["date", "source", "medium", "campaign", "term", "channel", "conversions"]],
        shape="row_level" if is_row_level else "aggregated_long",
        duplicates_removed=dups_removed,
        detected_columns={
            "timestamp": ts_col,
            "id": id_col,
            "source": src_col,
            "medium": med_col,
            "campaign": camp_col,
            "term": term_col,
            "channel": chan_col,
            "conversions": conv_col,
        },
    )


def load_evc(source: str | Path | pd.DataFrame, config: Config | None = None) -> pd.Series:
    """Load Google Ads daily Engaged-View Conversions into a daily float Series."""
    cfg = config or Config()
    ov: ColumnOverrides = cfg.columns
    df = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source, extra_header_tokens=("evc", "engaged"))

    cols = list(df.columns)
    date_col = _match_column(cols, TIMESTAMP_SYNONYMS, ov.evc_date)
    if date_col is None:
        raise ValueError(f"Could not detect date column in EVC file. Columns: {cols}")

    evc_col = _match_column(cols, EVC_SYNONYMS, ov.evc_column, exclude={date_col})
    if evc_col is None:
        # Fallback: single remaining numeric column
        remaining = [c for c in cols if c != date_col]
        if len(remaining) == 1:
            evc_col = remaining[0]
        else:
            raise ValueError(f"Could not detect EVC column in {cols}. Pass --evc-column.")

    dates = _parse_localized_dates(df[date_col])
    if dates.isna().any():
        fallback = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        dates = dates.fillna(fallback)
    dates = dates.dt.normalize()
    vals = _clean_numeric_series(df[evc_col]).fillna(0.0)
    clean = pd.DataFrame({"date": dates, "evc": vals}).dropna(subset=["date"])
    series = clean.groupby("date")["evc"].sum().sort_index().astype(float)
    series.name = "evc"
    return series


def load_controls(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load optional daily control columns (date + numeric covariates)."""
    df = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source)
    cols = list(df.columns)
    date_col = _match_column(cols, TIMESTAMP_SYNONYMS)
    if date_col is None:
        raise ValueError(f"Could not detect date column in controls file. Columns: {cols}")
    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed").dt.normalize()
    out = pd.DataFrame(index=dates)
    for c in cols:
        if c == date_col:
            continue
        out[c] = _clean_numeric_series(df[c])
    out = out.loc[out.index.notna()].groupby(level=0).mean().sort_index()
    if out.isna().any().any():
        raise ValueError("Controls file contains non-numeric or missing values.")
    return out
