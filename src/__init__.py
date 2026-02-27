"""Gold analysis package."""

from .html_charts import export_html_charts
from .model import train_and_forecast
from .scraper import fetch_gold_data, fetch_gold_history
from .utils import compatible_boxplot, generate_report, plot_results

__all__ = [
    "export_html_charts",
    "train_and_forecast",
    "fetch_gold_data",
    "fetch_gold_history",
    "compatible_boxplot",
    "generate_report",
    "plot_results",
]
