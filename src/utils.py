from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

_PERSIAN_DIGITS_MAP = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def configure_plot_style(language: str = "fa") -> str:
    """
    Configure plotting style and return effective language used for labels.
    It falls back to finglish when proper Persian-capable fonts are unavailable.
    """
    if language == "finglish":
        plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Tahoma"]
        plt.rcParams["axes.unicode_minus"] = False
        return "finglish"

    preferred_fonts = [
        "Vazirmatn",
        "Vazir",
        "IRANSans",
        "B Nazanin",
        "Noto Naskh Arabic",
        "Noto Sans Arabic",
        "Tahoma",
        "Segoe UI",
    ]
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_fonts = [name for name in preferred_fonts if name in installed_fonts]

    if not selected_fonts:
        # No Persian/Arabic-capable font in the environment.
        # Use finglish labels to prevent missing glyph warnings.
        plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Tahoma"]
        plt.rcParams["axes.unicode_minus"] = False
        return "finglish"

    warnings.filterwarnings(
        "ignore",
        message="Matplotlib currently does not support Arabic natively.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Glyph .* missing from font",
        category=UserWarning,
    )

    plt.rcParams["font.family"] = selected_fonts
    plt.rcParams["axes.unicode_minus"] = False
    return "fa"


def configure_matplotlib_for_farsi() -> None:
    configure_plot_style(language="fa")


def compatible_boxplot(ax: Any, data: Any, labels: list[str] | None = None, **kwargs: Any) -> Any:
    """
    Matplotlib compatibility wrapper:
    - Matplotlib < 3.9: uses `labels`
    - Matplotlib >= 3.9: uses `tick_labels`
    """
    if labels is None:
        return ax.boxplot(data, **kwargs)

    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


def to_persian_digits(value: str) -> str:
    return value.translate(_PERSIAN_DIGITS_MAP)


def format_number(value: float, digits: int = 0, use_persian_digits: bool = True) -> str:
    formatted = f"{value:,.{digits}f}"
    return to_persian_digits(formatted) if use_persian_digits else formatted


def format_date(value: pd.Timestamp, use_persian_digits: bool = True) -> str:
    if pd.isna(value):
        return "invalid"
    text = value.strftime("%Y-%m-%d")
    return to_persian_digits(text) if use_persian_digits else text


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "price"])

    columns = {str(col).strip().lower(): col for col in df.columns}
    date_col = None
    price_col = None

    if "date" in columns and "price" in columns:
        date_col = columns["date"]
        price_col = columns["price"]
    elif "ds" in columns and "y" in columns:
        date_col = columns["ds"]
        price_col = columns["y"]
    elif len(df.columns) >= 2:
        date_col = df.columns[0]
        price_col = df.columns[1]

    if date_col is None or price_col is None:
        return pd.DataFrame(columns=["date", "price"])

    clean = df[[date_col, price_col]].copy()
    clean.columns = ["date", "price"]

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["price"] = (
        clean["price"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    clean = clean.dropna(subset=["date", "price"])
    clean = clean[clean["price"] > 0]
    clean = clean.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    clean = _drop_return_outliers(clean, value_col="price")
    clean = clean.reset_index(drop=True)
    return clean


def read_price_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"فایل داده پیدا نشد: {path}")

    raw = pd.read_csv(path)
    normalized = normalize_price_frame(raw)
    if normalized.empty:
        raise ValueError("CSV معتبر نیست یا داده قابل استفاده ندارد.")
    return normalized


def save_price_csv(df: pd.DataFrame, path: Path) -> None:
    normalized = normalize_price_frame(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False, encoding="utf-8-sig")


def generate_sample_data(days: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    trend = np.linspace(25_000_000, 38_000_000, days)
    weekly = 260_000 * np.sin((2 * np.pi * np.arange(days)) / 7)
    monthly = 420_000 * np.sin((2 * np.pi * np.arange(days)) / 30)
    noise = rng.normal(loc=0, scale=120_000, size=days)

    prices = np.maximum(trend + weekly + monthly + noise, 1_000).round(0)
    return pd.DataFrame({"date": dates, "price": prices})


def plot_results(
    forecast: pd.DataFrame,
    historical_df: pd.DataFrame,
    save_path: Path | None = None,
    show: bool = True,
    language: str = "fa",
) -> None:
    effective_language = configure_plot_style(language=language)

    hist = normalize_price_frame(historical_df)
    pred = forecast.copy()

    pred["ds"] = pd.to_datetime(pred["ds"], errors="coerce")
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    pred = pred.dropna(subset=["ds", "yhat", "yhat_lower", "yhat_upper"])

    if hist.empty or pred.empty:
        return

    labels = _build_plot_labels(effective_language)

    plt.figure(figsize=(14, 7))
    plt.plot(hist["date"], hist["price"], label=labels["history"], color="#1f77b4", linewidth=2)
    plt.plot(pred["ds"], pred["yhat"], label=labels["forecast"], color="#2ca02c", linestyle="--", linewidth=2)
    plt.fill_between(
        pred["ds"],
        pred["yhat_lower"],
        pred["yhat_upper"],
        color="#7f7f7f",
        alpha=0.25,
        label=labels["interval"],
    )

    plt.title(labels["title"])
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    backend = str(plt.get_backend()).lower()
    if show and "agg" not in backend:
        plt.show()
    else:
        plt.close()


def generate_report(
    forecast: pd.DataFrame,
    historical_df: pd.DataFrame,
    diagnostics: dict[str, Any] | None,
    language: str = "fa",
) -> str:
    hist = normalize_price_frame(historical_df)
    if hist.empty or forecast.empty:
        return (
            "Gozaresh tolid nashod: dade kafi vojud nadarad."
            if language == "finglish"
            else "گزارش تولید نشد: داده کافی وجود ندارد."
        )

    last_actual_date = pd.to_datetime(hist["date"].iloc[-1], errors="coerce")
    last_actual_price = float(hist["price"].iloc[-1])

    first_pred = forecast.iloc[0]
    final_pred = forecast.iloc[-1]

    start_price = float(first_pred["yhat"])
    end_price = float(final_pred["yhat"])
    trend_label_fa = "صعودی" if end_price > start_price else "نزولی" if end_price < start_price else "خنثی"
    trend_label_finglish = "Soodi" if end_price > start_price else "Nozooli" if end_price < start_price else "Khonsa"

    model_name_map_fa = {
        "weighted_ensemble": "ترکیب وزنی مدل‌ها",
        "naive": "آخرین مقدار (Naive)",
        "drift": "روند خطی (Drift)",
        "seasonal_naive": "فصلی ساده (Seasonal Naive)",
        "theta_like": "مدل تتای ساده (Theta-like)",
        "holt_linear": "هموارسازی نمایی هولت (Holt Linear)",
        "hw_add": "هالت-وینتر جمعی (HW Additive)",
        "hw_damped": "هالت-وینتر دمپ‌شده (HW Damped)",
    }
    model_name_map_finglish = {
        "weighted_ensemble": "Tarkib Vazni Modelha (Weighted Ensemble)",
        "naive": "Akharin Meghdar (Naive)",
        "drift": "Ravand Khati (Drift)",
        "seasonal_naive": "Fasli Sade (Seasonal Naive)",
        "theta_like": "Theta Like",
        "holt_linear": "Hamvarsazi Holt (Holt Linear)",
        "hw_add": "Holt Winters Additive",
        "hw_damped": "Holt Winters Damped",
    }

    use_persian_digits = language == "fa"
    unknown_text = "نامشخص" if language == "fa" else "Unknown"

    selected_model = unknown_text
    base_best_model = unknown_text
    metrics_text = "در دسترس نیست" if language == "fa" else "Not available"
    holdout_text = unknown_text
    cv_text = unknown_text
    ensemble_text = ""

    if diagnostics:
        selected_key = str(diagnostics.get("selected_model", ""))
        base_key = str(diagnostics.get("base_best_model", selected_key))

        if language == "finglish":
            selected_model = model_name_map_finglish.get(selected_key, selected_key or unknown_text)
            base_best_model = model_name_map_finglish.get(base_key, base_key or unknown_text)
        else:
            selected_model = model_name_map_fa.get(selected_key, selected_key or unknown_text)
            base_best_model = model_name_map_fa.get(base_key, base_key or unknown_text)

        holdout_size = diagnostics.get("holdout_size")
        if isinstance(holdout_size, int):
            holdout_text = to_persian_digits(str(holdout_size)) if use_persian_digits else str(holdout_size)

        cv_splits = diagnostics.get("cv_splits")
        if isinstance(cv_splits, int):
            cv_text = to_persian_digits(str(cv_splits)) if use_persian_digits else str(cv_splits)

        model_scores = diagnostics.get("models", {})
        selected_metrics = model_scores.get(base_key, {}) if isinstance(model_scores, dict) else {}
        if selected_metrics:
            mae = float(selected_metrics.get("mae", np.nan))
            rmse = float(selected_metrics.get("rmse", np.nan))
            mape = float(selected_metrics.get("mape", np.nan))
            metrics_text = (
                f"MAE: {format_number(mae, 0, use_persian_digits=use_persian_digits)} | "
                f"RMSE: {format_number(rmse, 0, use_persian_digits=use_persian_digits)} | "
                f"MAPE: {format_number(mape, 2, use_persian_digits=use_persian_digits)}%"
            )

        members = diagnostics.get("ensemble_members", [])
        weights = diagnostics.get("ensemble_weights", {})
        if isinstance(members, list) and members and isinstance(weights, dict):
            readable_members = []
            for name in members:
                key = str(name)
                label = model_name_map_finglish.get(key, key) if language == "finglish" else model_name_map_fa.get(key, key)
                weight = float(weights.get(key, 0.0))
                readable_members.append(
                    f"{label}: {format_number(weight * 100, 1, use_persian_digits=use_persian_digits)}%"
                )
            joiner = " | "
            ensemble_text = joiner.join(readable_members)

    first_date = format_date(pd.to_datetime(first_pred["ds"]), use_persian_digits=use_persian_digits)
    final_date = format_date(pd.to_datetime(final_pred["ds"]), use_persian_digits=use_persian_digits)
    first_price = format_number(float(first_pred["yhat"]), 0, use_persian_digits=use_persian_digits)
    final_price = format_number(float(final_pred["yhat"]), 0, use_persian_digits=use_persian_digits)
    lower_text = format_number(float(final_pred["yhat_lower"]), 0, use_persian_digits=use_persian_digits)
    upper_text = format_number(float(final_pred["yhat_upper"]), 0, use_persian_digits=use_persian_digits)
    actual_date_text = format_date(last_actual_date, use_persian_digits=use_persian_digits)
    actual_price_text = format_number(last_actual_price, 0, use_persian_digits=use_persian_digits)

    if language == "finglish":
        lines = [
            "----- Gozaresh Tahlili Gheymat Tala -----",
            f"Tarikh akharin dade vaghei: {actual_date_text}",
            f"Akharin gheymat sabt shode: {actual_price_text}",
            f"Model montakhab: {selected_model}",
            f"Best model dar backtest: {base_best_model}",
            f"Andaze dade etebarsanji: {holdout_text} rooz",
            f"Tedad split backtest: {cv_text}",
            f"Khataye model montakhab: {metrics_text}",
        ]
        if ensemble_text:
            lines.append(f"Vazn modelha: {ensemble_text}")
        lines += [
            "",
            f"Pishbini rooz aval ({first_date}): {first_price}",
            f"Pishbini enteha-ye ofogh ({final_date}): {final_price}",
            f"Baze eteminan 95% dar enteha-ye ofogh: {lower_text} ta {upper_text}",
            f"Jahat ravand pishbini-shode: {trend_label_finglish}",
            "",
            "Hoshdar: in khoroji faghat amoozeshi ast va nabayad tanha mabnaye tasmim mali bashad.",
            "--------------------------------",
        ]
        return "\n".join(lines)

    lines = [
        "----- گزارش تحلیلی قیمت طلا -----",
        f"تاریخ آخرین داده واقعی: {actual_date_text}",
        f"آخرین قیمت ثبت‌شده: {actual_price_text}",
        f"مدل منتخب: {selected_model}",
        f"بهترین مدل در بک‌تست: {base_best_model}",
        f"اندازه داده اعتبارسنجی: {holdout_text} روز",
        f"تعداد اسپیلت بک‌تست: {cv_text}",
        f"خطای مدل منتخب: {metrics_text}",
    ]
    if ensemble_text:
        lines.append(f"وزن مدل‌ها: {ensemble_text}")

    lines += [
        "",
        f"پیش‌بینی روز اول ({first_date}): {first_price}",
        f"پیش‌بینی انتهای افق ({final_date}): {final_price}",
        f"بازه اطمینان ۹۵٪ در انتهای افق: {lower_text} تا {upper_text}",
        f"جهت روند پیش‌بینی‌شده: {trend_label_fa}",
        "",
        "هشدار: این خروجی صرفاً آموزشی است و نباید تنها مبنای تصمیم مالی قرار گیرد.",
        "--------------------------------",
    ]
    return "\n".join(lines)


def write_report(report_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8-sig")


def _drop_return_outliers(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Remove extreme one-step spikes by return, not by absolute level.
    This preserves long-term upward trends (e.g., inflation regime changes).
    """
    if len(df) < 20:
        return df

    view = df.copy()
    ret = view[value_col].pct_change()
    q1, q3 = ret.quantile([0.25, 0.75])
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 0:
        return df

    lower = q1 - (4 * iqr)
    upper = q3 + (4 * iqr)
    spike_mask = ret.between(lower, upper) | ret.isna()

    # Always keep recent tail to avoid cutting latest market regime.
    tail_keep = pd.Series(False, index=view.index)
    tail_keep.iloc[-10:] = True
    final_mask = spike_mask | tail_keep

    filtered = view[final_mask]
    if len(filtered) >= max(30, int(0.85 * len(df))):
        return filtered
    return df


def _build_plot_labels(language: str) -> dict[str, str]:
    if language == "finglish":
        return {
            "history": "Gheymat Tarikhi",
            "forecast": "Pishbini Model",
            "interval": "Baze Eteminan 95%",
            "title": "Tahlil va Pishbini Gheymat Tala",
            "x": "Tarikh",
            "y": "Gheymat",
        }

    return {
        "history": "قیمت تاریخی",
        "forecast": "پیش‌بینی مدل",
        "interval": "بازه اطمینان ۹۵٪",
        "title": "تحلیل و پیش‌بینی قیمت طلا",
        "x": "تاریخ",
        "y": "قیمت",
    }
