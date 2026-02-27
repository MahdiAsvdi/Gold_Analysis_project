from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import normalize_price_frame

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    px = None
    go = None
    make_subplots = None


def export_html_charts(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    diagnostics: dict[str, Any] | None,
    output_dir: Path,
    language: str = "finglish",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if px is None or go is None or make_subplots is None:
        fallback = output_dir / "dashboard.html"
        fallback.write_text(
            "<html><body><h2>Plotly not installed.</h2>"
            "<p>Please install requirements to generate interactive charts.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        return {"dashboard": fallback}

    hist = normalize_price_frame(historical_df)
    pred = _normalize_forecast_frame(forecast_df)

    labels = _labels(language)

    charts: dict[str, Path] = {}

    overview = _forecast_overview_chart(hist, pred, labels)
    charts["forecast_overview"] = output_dir / "01_forecast_overview.html"
    overview.write_html(charts["forecast_overview"], include_plotlyjs="cdn")

    ma_chart = _moving_average_chart(hist, pred, labels)
    charts["moving_averages"] = output_dir / "02_moving_averages.html"
    ma_chart.write_html(charts["moving_averages"], include_plotlyjs="cdn")

    returns_chart = _returns_chart(hist, labels)
    charts["returns_distribution"] = output_dir / "03_returns_distribution.html"
    returns_chart.write_html(charts["returns_distribution"], include_plotlyjs="cdn")

    weekday_chart = _weekday_pattern_chart(hist, labels)
    charts["weekday_pattern"] = output_dir / "04_weekday_pattern.html"
    weekday_chart.write_html(charts["weekday_pattern"], include_plotlyjs="cdn")

    model_chart = _model_comparison_chart(diagnostics, labels)
    charts["model_comparison"] = output_dir / "05_model_comparison.html"
    model_chart.write_html(charts["model_comparison"], include_plotlyjs="cdn")

    residual_chart = _residuals_chart(diagnostics, labels)
    charts["forecast_residuals"] = output_dir / "06_forecast_residuals.html"
    residual_chart.write_html(charts["forecast_residuals"], include_plotlyjs="cdn")

    dashboard = _build_dashboard(charts, output_dir, labels)
    charts["dashboard"] = dashboard
    return charts


def _normalize_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ds", "yhat", "yhat_lower", "yhat_upper"]).sort_values("ds")
    return out


def _forecast_overview_chart(hist: pd.DataFrame, pred: pd.DataFrame, labels: dict[str, str]):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["price"],
            mode="lines",
            name=labels["history"],
            line={"color": "#1f77b4", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred["ds"],
            y=pred["yhat"],
            mode="lines",
            name=labels["forecast"],
            line={"color": "#2ca02c", "width": 2, "dash": "dash"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pd.concat([pred["ds"], pred["ds"].iloc[::-1]], ignore_index=True),
            y=pd.concat([pred["yhat_upper"], pred["yhat_lower"].iloc[::-1]], ignore_index=True),
            fill="toself",
            fillcolor="rgba(127,127,127,0.20)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            name=labels["confidence"],
        )
    )

    if not pred.empty:
        fig.add_vline(x=pred["ds"].iloc[0], line_dash="dot", line_color="#666")

    fig.update_layout(
        title=labels["title_forecast"],
        xaxis_title=labels["x"],
        yaxis_title=labels["y"],
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    return fig


def _moving_average_chart(hist: pd.DataFrame, pred: pd.DataFrame, labels: dict[str, str]):
    view = hist.copy()
    view["ma7"] = view["price"].rolling(7).mean()
    view["ma30"] = view["price"].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["date"], y=view["price"], mode="lines", name=labels["history"]))
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma7"], mode="lines", name="MA-7", line={"color": "#ff7f0e"}))
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma30"], mode="lines", name="MA-30", line={"color": "#9467bd"}))

    if not pred.empty:
        fig.add_trace(
            go.Scatter(
                x=pred["ds"],
                y=pred["yhat"],
                mode="lines",
                name=labels["forecast"],
                line={"dash": "dash", "color": "#2ca02c"},
            )
        )

    fig.update_layout(
        title=labels["title_ma"],
        xaxis_title=labels["x"],
        yaxis_title=labels["y"],
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    return fig


def _returns_chart(hist: pd.DataFrame, labels: dict[str, str]):
    returns = hist.copy()
    returns["ret_pct"] = returns["price"].pct_change() * 100
    returns = returns.dropna(subset=["ret_pct"]) 

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=(labels["returns_line"], labels["returns_hist"]))

    fig.add_trace(
        go.Scatter(x=returns["date"], y=returns["ret_pct"], mode="lines", name=labels["returns"], line={"color": "#d62728"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=returns["ret_pct"], nbinsx=40, name=labels["returns_hist"], marker={"color": "#17becf"}),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=labels["title_returns"],
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    fig.update_xaxes(title_text=labels["x"], row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_xaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text=labels["count"], row=2, col=1)
    return fig


def _weekday_pattern_chart(hist: pd.DataFrame, labels: dict[str, str]):
    view = hist.copy()
    view["weekday"] = view["date"].dt.day_name()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig = px.box(
        view,
        x="weekday",
        y="price",
        category_orders={"weekday": order},
        points="outliers",
        color="weekday",
    )
    fig.update_layout(
        title=labels["title_weekday"],
        xaxis_title=labels["weekday"],
        yaxis_title=labels["y"],
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    return fig


def _model_comparison_chart(diagnostics: dict[str, Any] | None, labels: dict[str, str]):
    models = (diagnostics or {}).get("models", {})

    if not isinstance(models, dict) or not models:
        fig = go.Figure()
        fig.add_annotation(
            text=labels["no_model_data"],
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(template="plotly_white", title=labels["title_model_compare"])
        return fig

    rows = []
    for name, metric in models.items():
        rows.append(
            {
                "model": name,
                "mae": float(metric.get("mae", np.nan)),
                "rmse": float(metric.get("rmse", np.nan)),
                "mape": float(metric.get("mape", np.nan)),
            }
        )

    table = pd.DataFrame(rows).sort_values("mae")

    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "MAPE"))
    fig.add_trace(go.Bar(x=table["model"], y=table["mae"], marker={"color": "#1f77b4"}), row=1, col=1)
    fig.add_trace(go.Bar(x=table["model"], y=table["rmse"], marker={"color": "#ff7f0e"}), row=1, col=2)
    fig.add_trace(go.Bar(x=table["model"], y=table["mape"], marker={"color": "#2ca02c"}), row=1, col=3)

    fig.update_layout(
        title=labels["title_model_compare"],
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    return fig


def _residuals_chart(diagnostics: dict[str, Any] | None, labels: dict[str, str]):
    diag = diagnostics or {}
    errors_map = diag.get("model_errors", {})
    base_model = diag.get("base_best_model")

    residuals = []
    if isinstance(errors_map, dict) and base_model in errors_map:
        residuals = list(errors_map.get(base_model, []))

    fig = make_subplots(rows=1, cols=2, subplot_titles=(labels["residual_line"], labels["residual_hist"]))

    if residuals:
        idx = np.arange(1, len(residuals) + 1)
        fig.add_trace(go.Scatter(x=idx, y=residuals, mode="lines", line={"color": "#d62728"}), row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=35, marker={"color": "#8c564b"}), row=1, col=2)
    else:
        fig.add_annotation(text=labels["no_model_data"], x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

    fig.update_layout(
        title=labels["title_residuals"],
        template="plotly_white",
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    fig.update_xaxes(title_text=labels["index"], row=1, col=1)
    fig.update_yaxes(title_text=labels["residual"], row=1, col=1)
    fig.update_xaxes(title_text=labels["residual"], row=1, col=2)
    fig.update_yaxes(title_text=labels["count"], row=1, col=2)
    return fig


def _build_dashboard(charts: dict[str, Path], output_dir: Path, labels: dict[str, str]) -> Path:
    order = [
        "forecast_overview",
        "moving_averages",
        "returns_distribution",
        "weekday_pattern",
        "model_comparison",
        "forecast_residuals",
    ]

    tabs = []
    for key in order:
        if key in charts:
            tabs.append((key, charts[key].name))

    if not tabs:
        dashboard = output_dir / "dashboard.html"
        dashboard.write_text("<html><body>No charts generated.</body></html>", encoding="utf-8")
        return dashboard

    first_file = tabs[0][1]
    buttons_html = "\n".join(
        f"<button onclick=\"openChart('{name}')\">{_title_for_key(name, labels)}</button>" for name, _ in tabs
    )

    mapping_lines = ",\n".join(f"'{name}': '{filename}'" for name, filename in tabs)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{labels['dashboard_title']}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f5f7fb; }}
    header {{ background: #111827; color: #fff; padding: 14px 18px; }}
    .toolbar {{ padding: 12px 14px; display: flex; gap: 8px; flex-wrap: wrap; background: #e5e7eb; }}
    button {{ background: #1f2937; color: #fff; border: 0; border-radius: 7px; padding: 8px 12px; cursor: pointer; }}
    button:hover {{ background: #374151; }}
    iframe {{ width: 100%; height: calc(100vh - 110px); border: 0; }}
  </style>
</head>
<body>
  <header><h3>{labels['dashboard_title']}</h3></header>
  <div class=\"toolbar\">{buttons_html}</div>
  <iframe id=\"viewer\" src=\"{first_file}\"></iframe>
  <script>
    const chartMap = {{
      {mapping_lines}
    }};
    function openChart(name) {{
      const file = chartMap[name];
      if (file) {{
        document.getElementById('viewer').src = file;
      }}
    }}
  </script>
</body>
</html>
""".strip()

    dashboard = output_dir / "dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    return dashboard


def _title_for_key(key: str, labels: dict[str, str]) -> str:
    mapping = {
        "forecast_overview": labels["title_forecast"],
        "moving_averages": labels["title_ma"],
        "returns_distribution": labels["title_returns"],
        "weekday_pattern": labels["title_weekday"],
        "model_comparison": labels["title_model_compare"],
        "forecast_residuals": labels["title_residuals"],
    }
    return mapping.get(key, key)


def _labels(language: str) -> dict[str, str]:
    if language == "fa":
        return {
            "history": "قیمت تاریخی",
            "forecast": "پیش‌بینی",
            "confidence": "بازه اطمینان 95%",
            "title_forecast": "نمای کلی قیمت و پیش‌بینی",
            "title_ma": "میانگین‌های متحرک",
            "title_returns": "رفتار بازده روزانه",
            "title_weekday": "الگوی روزهای هفته",
            "title_model_compare": "مقایسه دقت مدل‌ها",
            "title_residuals": "تحلیل خطای پیش‌بینی",
            "dashboard_title": "داشبورد نمودارهای تحلیل طلا",
            "x": "تاریخ",
            "y": "قیمت",
            "returns": "بازده روزانه",
            "returns_line": "سری بازده",
            "returns_hist": "توزیع بازده",
            "count": "تعداد",
            "weekday": "روز هفته",
            "residual": "خطا",
            "residual_line": "سری خطا",
            "residual_hist": "توزیع خطا",
            "index": "اندیس",
            "no_model_data": "داده مدل در دسترس نیست",
        }

    return {
        "history": "Historical Price",
        "forecast": "Forecast",
        "confidence": "95% Interval",
        "title_forecast": "Price and Forecast Overview",
        "title_ma": "Moving Averages",
        "title_returns": "Daily Returns Analysis",
        "title_weekday": "Weekday Pattern",
        "title_model_compare": "Model Accuracy Comparison",
        "title_residuals": "Forecast Residual Diagnostics",
        "dashboard_title": "Gold Analysis Charts Dashboard",
        "x": "Date",
        "y": "Price",
        "returns": "Daily Return",
        "returns_line": "Return Series",
        "returns_hist": "Return Distribution",
        "count": "Count",
        "weekday": "Weekday",
        "residual": "Residual",
        "residual_line": "Residual Series",
        "residual_hist": "Residual Distribution",
        "index": "Index",
        "no_model_data": "No model diagnostics available",
    }
