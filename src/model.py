from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None


ModelFunc = Callable[[np.ndarray, int], np.ndarray]


def train_and_forecast(df: pd.DataFrame, periods: int = 30) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """
    Improved forecasting pipeline:
    - robust input cleaning
    - multi-split backtesting
    - weighted ensemble from top models
    - uncertainty from residuals + model spread
    """
    clean_df = _clean_input(df)
    if clean_df.empty or len(clean_df) < 35:
        return None, None

    horizon = max(int(periods), 1)
    series = clean_df["y"].to_numpy(dtype=float)

    holdout = _select_holdout_size(len(series))
    splits = _make_backtest_splits(series, holdout=holdout, max_splits=3, min_train=45)
    if not splits:
        return None, None

    season_length = _infer_season_length(clean_df["ds"])
    candidates = _build_model_candidates(season_length)

    scoreboard, errors_by_model = _evaluate_models(candidates, splits)
    if not scoreboard:
        return None, None

    ranked = sorted(scoreboard.items(), key=lambda kv: (kv[1]["mae"], kv[1]["rmse"]))
    best_model = ranked[0][0]
    top_count = min(3, len(ranked))
    ensemble_members = [name for name, _ in ranked[:top_count]]

    weights = _inverse_error_weights({name: scoreboard[name]["mae"] for name in ensemble_members})
    member_forecasts: dict[str, np.ndarray] = {}
    for name in ensemble_members:
        member_forecasts[name] = _safe_forecast(candidates[name], series, horizon)

    ensemble_forecast = np.zeros(horizon, dtype=float)
    for name, weight in weights.items():
        ensemble_forecast += member_forecasts[name] * weight
    ensemble_forecast = np.maximum(ensemble_forecast, 0)

    if len(ensemble_members) > 1:
        spread_matrix = np.vstack([member_forecasts[name] for name in ensemble_members])
    else:
        spread_matrix = np.empty((0, horizon), dtype=float)

    best_errors = errors_by_model.get(best_model, np.array([], dtype=float))
    yhat_lower, yhat_upper = _build_confidence_intervals(
        yhat=ensemble_forecast,
        validation_errors=best_errors,
        original_series=series,
        model_spread=spread_matrix,
    )

    last_date = clean_df["ds"].iloc[-1]
    freq = _infer_frequency(clean_df["ds"])
    future_dates = pd.date_range(start=last_date + freq, periods=horizon, freq=freq)

    forecast = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": ensemble_forecast,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
        }
    )

    diagnostics: dict[str, Any] = {
        "selected_model": "weighted_ensemble",
        "base_best_model": best_model,
        "ensemble_members": ensemble_members,
        "ensemble_weights": {k: float(v) for k, v in weights.items()},
        "holdout_size": holdout,
        "cv_splits": len(splits),
        "models": scoreboard,
        "points_used": len(series),
        "model_errors": {k: v.tolist() for k, v in errors_by_model.items()},
        "season_length": season_length,
    }
    return forecast, diagnostics


def _clean_input(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    columns = {str(c).strip().lower(): c for c in df.columns}

    if "ds" in columns and "y" in columns:
        ds_col = columns["ds"]
        y_col = columns["y"]
    elif "date" in columns and "price" in columns:
        ds_col = columns["date"]
        y_col = columns["price"]
    else:
        if len(df.columns) < 2:
            return pd.DataFrame(columns=["ds", "y"])
        ds_col = df.columns[0]
        y_col = df.columns[1]

    clean = df[[ds_col, y_col]].copy()
    clean.columns = ["ds", "y"]

    clean["ds"] = pd.to_datetime(clean["ds"], errors="coerce")
    clean["y"] = (
        clean["y"]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    clean = clean.dropna(subset=["ds", "y"]).sort_values("ds")
    clean = clean.drop_duplicates(subset=["ds"], keep="last")
    clean = clean[clean["y"] > 0]
    clean = _drop_return_outliers(clean, value_col="y")
    clean = clean.reset_index(drop=True)
    return clean


def _select_holdout_size(n_points: int) -> int:
    target = int(round(n_points * 0.16))
    return max(10, min(30, target, n_points - 10))


def _make_backtest_splits(
    series: np.ndarray,
    holdout: int,
    max_splits: int,
    min_train: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_points = len(series)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for offset in range(max_splits - 1, -1, -1):
        valid_end = n_points - (offset * holdout)
        valid_start = valid_end - holdout
        train_end = valid_start

        if valid_start <= 0 or valid_end > n_points:
            continue
        if train_end < min_train:
            continue

        train = series[:train_end]
        valid = series[valid_start:valid_end]
        if len(valid) == holdout:
            splits.append((train, valid))

    if not splits and n_points > (holdout + 7):
        splits = [(series[:-holdout], series[-holdout:])]
    return splits


def _build_model_candidates(season_length: int) -> dict[str, ModelFunc]:
    models: dict[str, ModelFunc] = {
        "naive": _forecast_naive,
        "drift": _forecast_drift,
        "theta_like": _forecast_theta_like,
        "holt_linear": _forecast_holt_auto,
    }

    if season_length >= 2:
        models["seasonal_naive"] = lambda s, h, m=season_length: _forecast_seasonal_naive(s, h, m)

    if ExponentialSmoothing is not None and season_length >= 2:
        models["hw_add"] = lambda s, h, m=season_length: _forecast_hw_add(s, h, m, damped=False)
        models["hw_damped"] = lambda s, h, m=season_length: _forecast_hw_add(s, h, m, damped=True)

    return models


def _evaluate_models(
    candidates: dict[str, ModelFunc],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    scoreboard: dict[str, dict[str, float]] = {}
    errors_by_model: dict[str, np.ndarray] = {}

    for model_name, forecaster in candidates.items():
        all_errors: list[np.ndarray] = []
        all_actuals: list[np.ndarray] = []

        for train, valid in splits:
            pred = _safe_forecast(forecaster, train, len(valid))
            errors = valid - pred
            all_errors.append(errors)
            all_actuals.append(valid)

        if not all_errors:
            continue

        err = np.concatenate(all_errors)
        actual = np.concatenate(all_actuals)

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = float(np.mean(np.abs(err) / np.clip(np.abs(actual), 1.0, None)) * 100)

        scoreboard[model_name] = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
        errors_by_model[model_name] = err

    return scoreboard, errors_by_model


def _inverse_error_weights(mae_by_model: dict[str, float]) -> dict[str, float]:
    if not mae_by_model:
        return {}

    raw = {name: 1.0 / max(mae, 1e-6) for name, mae in mae_by_model.items()}
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(raw)
        return {name: uniform for name in raw}
    return {name: value / total for name, value in raw.items()}


def _safe_forecast(forecaster: ModelFunc, series: np.ndarray, horizon: int) -> np.ndarray:
    try:
        pred = np.asarray(forecaster(series, horizon), dtype=float)
        if pred.shape[0] != horizon:
            raise ValueError("unexpected forecast length")
        if not np.all(np.isfinite(pred)):
            raise ValueError("forecast contains non-finite values")
        return np.maximum(pred, 0)
    except Exception:
        return _forecast_naive(series, horizon)


def _forecast_naive(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(series[-1], horizon).astype(float)


def _forecast_drift(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) < 2:
        return _forecast_naive(series, horizon)

    slope = (series[-1] - series[0]) / max(len(series) - 1, 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return series[-1] + (slope * steps)


def _forecast_seasonal_naive(series: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    if len(series) < season_length or season_length <= 1:
        return _forecast_naive(series, horizon)

    tail = series[-season_length:]
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(tail, repeats)[:horizon].astype(float)


def _forecast_theta_like(series: np.ndarray, horizon: int) -> np.ndarray:
    drift = _forecast_drift(series, horizon)
    ses = _forecast_ses_auto(series, horizon)
    return (0.45 * drift) + (0.55 * ses)


def _forecast_ses_auto(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) < 3:
        return _forecast_naive(series, horizon)

    alpha_grid = np.linspace(0.1, 0.9, 9)
    best_alpha = 0.3
    best_mae = float("inf")

    for alpha in alpha_grid:
        mae = _ses_one_step_mae(series, float(alpha))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    level = float(series[0])
    for value in series[1:]:
        level = (best_alpha * value) + ((1 - best_alpha) * level)

    return np.repeat(level, horizon).astype(float)


def _ses_one_step_mae(series: np.ndarray, alpha: float) -> float:
    level = float(series[0])
    errors = []

    for value in series[1:]:
        pred = level
        errors.append(abs(value - pred))
        level = (alpha * value) + ((1 - alpha) * level)

    return float(np.mean(errors))


def _forecast_holt_auto(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) < 3:
        return _forecast_drift(series, horizon)

    alpha_grid = np.linspace(0.1, 0.9, 9)
    beta_grid = np.linspace(0.1, 0.9, 9)

    best_alpha = 0.3
    best_beta = 0.1
    best_error = float("inf")

    for alpha in alpha_grid:
        for beta in beta_grid:
            error = _holt_one_step_mae(series, float(alpha), float(beta))
            if error < best_error:
                best_error = error
                best_alpha = float(alpha)
                best_beta = float(beta)

    return _forecast_holt(series, horizon, alpha=best_alpha, beta=best_beta)


def _holt_one_step_mae(series: np.ndarray, alpha: float, beta: float) -> float:
    if len(series) < 3:
        return float("inf")

    level = float(series[0])
    trend = float(series[1] - series[0])
    errors = []

    for value in series[1:]:
        pred = level + trend
        errors.append(abs(value - pred))

        prev_level = level
        level = (alpha * value) + ((1 - alpha) * (level + trend))
        trend = (beta * (level - prev_level)) + ((1 - beta) * trend)

    return float(np.mean(errors))


def _forecast_holt(series: np.ndarray, horizon: int, alpha: float, beta: float) -> np.ndarray:
    level = float(series[0])
    trend = float(series[1] - series[0])

    for value in series[1:]:
        prev_level = level
        level = (alpha * value) + ((1 - alpha) * (level + trend))
        trend = (beta * (level - prev_level)) + ((1 - beta) * trend)

    steps = np.arange(1, horizon + 1, dtype=float)
    return level + (trend * steps)


def _forecast_hw_add(series: np.ndarray, horizon: int, season_length: int, damped: bool) -> np.ndarray:
    if ExponentialSmoothing is None:
        return _forecast_holt_auto(series, horizon)

    min_len = max(season_length * 2, 20)
    if len(series) < min_len:
        return _forecast_holt_auto(series, horizon)

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=season_length,
        damped_trend=damped,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, use_brute=False)
    forecast = fitted.forecast(horizon)
    return np.asarray(forecast, dtype=float)


def _build_confidence_intervals(
    yhat: np.ndarray,
    validation_errors: np.ndarray,
    original_series: np.ndarray,
    model_spread: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(validation_errors) > 1:
        sigma = float(np.std(validation_errors, ddof=1))
    else:
        sigma = float(np.std(np.diff(original_series))) if len(original_series) > 1 else 0.0

    sigma = max(sigma, 1.0)
    steps = np.sqrt(np.arange(1, len(yhat) + 1, dtype=float))

    if model_spread.size > 0:
        spread = np.std(model_spread, axis=0)
    else:
        spread = np.zeros_like(yhat)

    margin = (1.96 * sigma * steps) + (1.15 * spread)
    lower = np.maximum(yhat - margin, 0)
    upper = yhat + margin
    return lower, upper


def _infer_season_length(ds: pd.Series) -> int:
    inferred = pd.infer_freq(ds)
    if inferred:
        if inferred.startswith("D"):
            return 7
        if inferred.startswith("W"):
            return 4
        if inferred.startswith("M"):
            return 12

    diffs = ds.sort_values().diff().dropna()
    if diffs.empty:
        return 7

    median_days = float(diffs.dt.total_seconds().median() / 86400.0)
    if median_days <= 2:
        return 7
    if median_days <= 10:
        return 4
    if median_days <= 40:
        return 12
    return 0


def _infer_frequency(ds: pd.Series) -> pd.DateOffset | pd.Timedelta:
    inferred = pd.infer_freq(ds)
    if inferred:
        return pd.tseries.frequencies.to_offset(inferred)

    diffs = ds.sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)

    mode = diffs.mode()
    if mode.empty:
        return pd.Timedelta(days=1)
    return mode.iloc[0]


def _drop_return_outliers(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
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

    tail_keep = pd.Series(False, index=view.index)
    tail_keep.iloc[-10:] = True
    final_mask = spike_mask | tail_keep

    filtered = view[final_mask]
    if len(filtered) >= max(30, int(0.85 * len(df))):
        return filtered
    return df
