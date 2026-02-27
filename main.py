from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.html_charts import export_html_charts
from src.model import train_and_forecast
from src.scraper import fetch_gold_data, fetch_gold_history
from src.utils import (
    configure_plot_style,
    generate_report,
    plot_results,
    save_price_csv,
    write_report,
)


def _ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="تحلیل و پیش‌بینی قیمت طلا")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/gold_prices.csv"),
        help="مسیر فایل CSV با ستون‌های date/price",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=30,
        help="تعداد روزهای پیش‌بینی",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/analysis_report.txt"),
        help="مسیر ذخیره گزارش متنی",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("data/forecast_plot.png"),
        help="مسیر ذخیره نمودار",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="اگر فعال شود، نمودار نمایش داده نمی‌شود",
    )
    parser.add_argument(
        "--refresh-live-price",
        action="store_true",
        help="برای گرفتن آخرین قیمت آنلاین و افزودن به CSV",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("data/html_charts"),
        help="مسیر ذخیره نمودارهای HTML",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="اگر فعال شود خروجی HTML تولید نمی‌شود",
    )
    parser.add_argument(
        "--language",
        choices=["fa", "finglish"],
        default="finglish",
        help="زبان خروجی گزارش و نمودار",
    )
    args = parser.parse_args()
    if args.forecast_days < 1:
        parser.error("--forecast-days باید عددی بزرگ‌تر از صفر باشد.")
    return args


def _load_historical_data(csv_path: Path, language: str) -> pd.DataFrame:
    tgju_data = fetch_gold_history(slug="geram18", unit="toman")
    if tgju_data is not None and not tgju_data.empty:
        save_price_csv(tgju_data, csv_path)
        print(_msg("tgju_history_loaded", language, rows=str(len(tgju_data))))
        return tgju_data

    print(_msg("tgju_history_failed", language))
    raise RuntimeError(_msg("no_data_available", language))


def _refresh_latest_price(history: pd.DataFrame, csv_path: Path, language: str) -> pd.DataFrame:
    live_row = fetch_gold_data(slug="geram18", unit="toman")
    if live_row is None:
        print(_msg("live_fetch_failed", language))
        return history

    adjusted_price, scale_note = _align_price_scale(
        new_price=float(live_row["price"]),
        history_prices=history["price"] if "price" in history.columns else pd.Series(dtype=float),
    )
    live_row["price"] = adjusted_price

    if scale_note:
        print(_msg("scale_adjusted", language, note=scale_note))

    history_price = pd.to_numeric(history["price"], errors="coerce").dropna()
    if history_price.empty:
        return history

    last_price = float(history_price.iloc[-1])
    change_ratio = abs(adjusted_price - last_price) / max(last_price, 1.0)
    if change_ratio > 0.35:
        print(_msg("outlier_live_price", language))
        return history

    new_row = pd.DataFrame([live_row])
    new_row["date"] = pd.to_datetime(new_row["date"], errors="coerce")
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    merged = (
        pd.concat([history, new_row], ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    save_price_csv(merged, csv_path)
    print(_msg("live_price_added", language))
    return merged


def _align_price_scale(new_price: float, history_prices: pd.Series) -> tuple[float, str | None]:
    history_numeric = pd.to_numeric(history_prices, errors="coerce").dropna()
    if history_numeric.empty:
        return new_price, None

    median_price = float(history_numeric.median())
    if median_price <= 0:
        return new_price, None

    ratio = new_price / median_price
    if 8 <= ratio <= 12:
        return new_price / 10.0, "rial_to_toman"
    if 0.08 <= ratio <= 0.12:
        return new_price * 10.0, "toman_to_rial"
    return new_price, None


def _msg(key: str, language: str, **kwargs: str) -> str:
    table = {
        "start": {
            "fa": "شروع تحلیل قیمت طلا...",
            "finglish": "Shoroo tahlil gheymat tala...",
        },
        "tgju_history_loaded": {
            "fa": "تاریخچه TGJU دریافت شد (تعداد رکورد: {rows}). مبنا: تومان.",
            "finglish": "TGJU history daryaft shod (rows: {rows}). Mabna: toman.",
        },
        "tgju_history_failed": {
            "fa": "دریافت تاریخچه از TGJU ناموفق بود.",
            "finglish": "Daryaft history az TGJU namovafagh bood.",
        },
        "no_data_available": {
            "fa": "هیچ داده معتبری از TGJU دریافت نشد. اتصال اینترنت یا دسترسی TGJU را بررسی کنید.",
            "finglish": "Hich dade motabari az TGJU daryaft nashod. Ettesal internet ya dastresi TGJU ra barrasi konid.",
        },
        "csv_read_failed": {
            "fa": "خواندن CSV موفق نبود ({error}).",
            "finglish": "CSV read movafagh nabood ({error}).",
        },
        "live_fetch_failed": {
            "fa": "گرفتن قیمت آنلاین موفق نبود. اجرای تحلیل با داده موجود ادامه دارد.",
            "finglish": "Gereftan gheymat online movafagh nabood. Tahlil ba dade mojood edame darad.",
        },
        "scale_adjusted": {
            "fa": "مقیاس قیمت آنلاین اصلاح شد ({note}).",
            "finglish": "Meghyas gheymat online eslah shod ({note}).",
        },
        "outlier_live_price": {
            "fa": "قیمت آنلاین پرت تشخیص داده شد و به دیتاست اضافه نشد.",
            "finglish": "Gheymat online outlier tashkhis dade shod va be dataset ezafe nashod.",
        },
        "live_price_added": {
            "fa": "آخرین قیمت آنلاین به دیتاست اضافه شد.",
            "finglish": "Akharin gheymat online be dataset ezafe shod.",
        },
        "model_failed": {
            "fa": "مدل‌سازی انجام نشد. داده‌ها را بررسی کنید.",
            "finglish": "Modelsazi anjam nashod. Dadeha ra barrasi konid.",
        },
        "html_done": {
            "fa": "خروجی HTML ساخته شد: {path}",
            "finglish": "HTML output sakhte shod: {path}",
        },
        "html_failed": {
            "fa": "تولید HTML موفق نبود: {error}",
            "finglish": "Tolid HTML movafagh nabood: {error}",
        },
    }

    scale_notes = {
        "rial_to_toman": {
            "fa": "تبدیل از ریال به تومان",
            "finglish": "tabdil az rial be toman",
        },
        "toman_to_rial": {
            "fa": "تبدیل از تومان به ریال",
            "finglish": "tabdil az toman be rial",
        },
    }

    if "note" in kwargs and kwargs["note"] in scale_notes:
        kwargs["note"] = scale_notes[kwargs["note"]].get(language, kwargs["note"])

    entry = table.get(key, {})
    template = entry.get(language, entry.get("fa", key))
    return template.format(**kwargs)


def main() -> None:
    _ensure_utf8_output()
    args = _parse_args()

    print(_msg("start", args.language))
    try:
        historical_data = _load_historical_data(args.csv, args.language)
    except RuntimeError as exc:
        print(str(exc))
        return

    if args.refresh_live_price:
        historical_data = _refresh_latest_price(historical_data, args.csv, args.language)

    forecast, diagnostics = train_and_forecast(historical_data, periods=args.forecast_days)
    if forecast is None:
        print(_msg("model_failed", args.language))
        return

    report_text = generate_report(
        forecast,
        historical_data,
        diagnostics,
        language=args.language,
    )
    print(report_text)
    write_report(report_text, args.report_path)

    if not args.no_plot:
        configure_plot_style(language=args.language)
        plot_results(
            forecast,
            historical_data,
            save_path=args.plot_path,
            show=True,
            language=args.language,
        )

    if not args.no_html:
        try:
            chart_paths = export_html_charts(
                historical_df=historical_data,
                forecast_df=forecast,
                diagnostics=diagnostics,
                output_dir=args.html_dir,
                language=args.language,
            )
            dashboard_path = chart_paths.get("dashboard")
            if dashboard_path is not None:
                print(_msg("html_done", args.language, path=str(dashboard_path)))
        except Exception as exc:
            print(_msg("html_failed", args.language, error=str(exc)))


if __name__ == "__main__":
    main()
