from __future__ import annotations

import random
import re
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

_TGJU_HISTORY_API = "https://api.tgju.org/v1/market/indicator/summary-table-data/{slug}?lang=fa&order_dir=asc"
_TGJU_PROFILE_URL = "https://www.tgju.org/profile/{slug}"

_PERSIAN_TO_EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def fetch_gold_history(
    slug: str = "geram18",
    unit: str = "toman",
    timeout: int = 20,
    retries: int = 3,
    max_rows: int | None = None,
) -> pd.DataFrame | None:
    """
    Fetch full historical price series from TGJU API.

    Parameters:
    - slug: instrument slug on TGJU (default: geram18)
    - unit: "toman" (default) or "rial"
    - max_rows: optional cap from newest side after cleaning

    Returns DataFrame columns: date, price
    """
    session = requests.Session()
    last_error: Exception | None = None
    url = _TGJU_HISTORY_API.format(slug=slug)

    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                time.sleep(random.uniform(1.0, 2.5))

            response = session.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/json,text/plain,*/*",
                },
                timeout=timeout,
            )
            response.raise_for_status()

            payload = response.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list) or not rows:
                raise ValueError("TGJU history API returned empty data")

            records: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, list) or len(row) < 7:
                    continue

                # Column index 3 is the close/last value in the summary table rows.
                close_raw = row[3]
                date_raw = row[6]

                date_value = _parse_gregorian_date(date_raw)
                price_value = _parse_numeric_text(str(close_raw))

                if date_value is None or price_value is None:
                    continue

                if unit == "toman":
                    price_value = price_value / 10.0

                records.append({"date": date_value, "price": float(price_value)})

            if not records:
                raise ValueError("No valid historical rows parsed from TGJU")

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.dropna(subset=["date", "price"])
            df = df[df["price"] > 0]
            df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            df = df.reset_index(drop=True)

            if max_rows is not None and max_rows > 0 and len(df) > max_rows:
                df = df.iloc[-max_rows:].reset_index(drop=True)

            if df.empty:
                raise ValueError("Final historical DataFrame is empty")

            return df

        except Exception as exc:
            last_error = exc

    print(f"خطا در دریافت تاریخچه TGJU: {last_error}")
    return None


def fetch_gold_data(
    slug: str = "geram18",
    unit: str = "toman",
    timeout: int = 12,
    retries: int = 3,
) -> dict[str, Any] | None:
    """
    Fetch latest TGJU price row. Prefer API history endpoint, fallback to profile page parse.
    Returns: {"date": "YYYY-MM-DD", "price": float}
    """
    history = fetch_gold_history(slug=slug, unit=unit, timeout=timeout, retries=retries, max_rows=5)
    if history is not None and not history.empty:
        last = history.iloc[-1]
        return {
            "date": pd.to_datetime(last["date"]).strftime("%Y-%m-%d"),
            "price": float(last["price"]),
        }

    return _fetch_latest_from_profile(slug=slug, unit=unit, timeout=timeout, retries=retries)


def _fetch_latest_from_profile(
    slug: str,
    unit: str,
    timeout: int,
    retries: int,
) -> dict[str, Any] | None:
    profile_url = _TGJU_PROFILE_URL.format(slug=slug)
    session = requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                time.sleep(random.uniform(1.0, 2.5))

            response = session.get(
                profile_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
                timeout=timeout,
            )
            response.raise_for_status()

            price = _extract_price_from_profile_html(response.text)
            if price is None:
                raise ValueError("عدد قیمت در صفحه پیدا نشد.")

            if unit == "toman":
                price = price / 10.0

            today = datetime.now().strftime("%Y-%m-%d")
            return {"date": today, "price": float(price)}

        except Exception as exc:
            last_error = exc

    print(f"خطا در استخراج قیمت آنلاین: {last_error}")
    return None


def _extract_price_from_profile_html(html_text: str) -> float | None:
    soup = BeautifulSoup(html_text, "html.parser")

    candidate_selectors = [
        "span[data-col='info.last_trade.PDrCotVal']",
        "span[data-col='info.last_trade.value']",
        "span.value",
        "td[data-col='info.last_trade.PDrCotVal']",
        "td[data-col='info.last_trade.value']",
    ]

    for selector in candidate_selectors:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            parsed = _parse_numeric_text(node.get_text(strip=True))
            if parsed is not None:
                return parsed

    full_text = soup.get_text(" ", strip=True)
    probable_values = re.findall(r"\b\d{1,3}(?:[,\u066C]\d{3})+(?:\.\d+)?\b", full_text)

    parsed_numbers = []
    for value in probable_values:
        parsed = _parse_numeric_text(value)
        if parsed is not None:
            parsed_numbers.append(parsed)

    if not parsed_numbers:
        return None

    parsed_numbers.sort()
    return parsed_numbers[-1]


def _parse_gregorian_date(value: Any) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip().translate(_PERSIAN_TO_EN_DIGITS)
    text = text.replace("-", "/")

    for fmt in ("%Y/%m/%d", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        dt = pd.to_datetime(text, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _parse_numeric_text(text: str) -> float | None:
    if not text:
        return None

    normalized = (
        text.translate(_PERSIAN_TO_EN_DIGITS)
        .replace("٬", ",")
        .replace("ریال", "")
        .replace("تومان", "")
    )
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = re.sub(r"[^0-9,\.\-]", "", normalized)

    if not normalized:
        return None

    normalized = normalized.replace(",", "")

    try:
        value = float(normalized)
    except ValueError:
        return None

    if value <= 0:
        return None
    return value
