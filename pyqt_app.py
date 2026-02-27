from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from PyQt5.QtCore import Qt, QUrl
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyQt5 + PyQtWebEngine nasb نیست. Lotfan requirements ra نصب کن. "
        f"Details: {exc}"
    )

from src.html_charts import export_html_charts
from src.model import train_and_forecast
from src.scraper import fetch_gold_data, fetch_gold_history
from src.utils import (
    generate_report,
    save_price_csv,
    write_report,
)


class GoldAnalysisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gold Analysis - PyQt")
        self.resize(1450, 880)

        self._dashboard_path: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        controls = self._build_controls()
        root_layout.addLayout(controls)

        splitter = QSplitter()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.report_box = QPlainTextEdit()
        self.report_box.setReadOnly(True)
        left_layout.addWidget(QLabel("Report"))
        left_layout.addWidget(self.report_box)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.chart_tabs = QTabWidget()
        right_layout.addWidget(self.chart_tabs)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([420, 1000])

        root_layout.addWidget(splitter)

        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    def _build_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.csv_input = QLineEdit(str(Path("data/gold_prices.csv")))
        self.csv_input.setMinimumWidth(300)

        self.csv_btn = QPushButton("CSV")
        self.csv_btn.clicked.connect(self._browse_csv)

        self.html_dir_input = QLineEdit(str(Path("data/html_charts")))
        self.html_dir_input.setMinimumWidth(300)

        self.html_btn = QPushButton("HTML Dir")
        self.html_btn.clicked.connect(self._browse_html_dir)

        self.forecast_spin = QSpinBox()
        self.forecast_spin.setMinimum(1)
        self.forecast_spin.setMaximum(365)
        self.forecast_spin.setValue(30)

        self.language_combo = QComboBox()
        self.language_combo.addItem("finglish")
        self.language_combo.addItem("fa")

        self.refresh_live_check = QCheckBox("Refresh Live Price")

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._run_analysis)

        self.open_dashboard_btn = QPushButton("Open Dashboard Tab")
        self.open_dashboard_btn.clicked.connect(self._open_dashboard_tab)

        layout.addWidget(QLabel("CSV:"))
        layout.addWidget(self.csv_input)
        layout.addWidget(self.csv_btn)

        layout.addWidget(QLabel("Forecast Days:"))
        layout.addWidget(self.forecast_spin)

        layout.addWidget(QLabel("Language:"))
        layout.addWidget(self.language_combo)

        layout.addWidget(self.refresh_live_check)

        layout.addWidget(QLabel("HTML:"))
        layout.addWidget(self.html_dir_input)
        layout.addWidget(self.html_btn)

        layout.addWidget(self.run_btn)
        layout.addWidget(self.open_dashboard_btn)
        return layout

    def _browse_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if file_path:
            self.csv_input.setText(file_path)

    def _browse_html_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select HTML Output Directory", str(Path.cwd()))
        if folder:
            self.html_dir_input.setText(folder)

    def _run_analysis(self) -> None:
        csv_path = Path(self.csv_input.text().strip())
        html_dir = Path(self.html_dir_input.text().strip())
        forecast_days = int(self.forecast_spin.value())
        language = self.language_combo.currentText().strip()

        try:
            self.statusBar().showMessage("Running analysis...")
            QApplication.setOverrideCursor(Qt.WaitCursor)

            history = self._load_historical_data(csv_path)
            if self.refresh_live_check.isChecked():
                history = self._refresh_latest_price(history, csv_path)

            forecast, diagnostics = train_and_forecast(history, periods=forecast_days)
            if forecast is None:
                raise RuntimeError("Model output is empty. Need more clean data.")

            report = generate_report(
                forecast=forecast,
                historical_df=history,
                diagnostics=diagnostics,
                language=language,
            )
            self.report_box.setPlainText(report)

            report_path = html_dir / "analysis_report.txt"
            write_report(report, report_path)

            chart_paths = export_html_charts(
                historical_df=history,
                forecast_df=forecast,
                diagnostics=diagnostics,
                output_dir=html_dir,
                language=language,
            )
            self._dashboard_path = chart_paths.get("dashboard")
            self._load_chart_tabs(chart_paths)

            self.statusBar().showMessage(f"Done. Charts saved in: {html_dir}")
        except Exception as exc:
            self.statusBar().showMessage("Error")
            QMessageBox.critical(self, "Analysis Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _open_dashboard_tab(self) -> None:
        if self._dashboard_path is None or not self._dashboard_path.exists():
            QMessageBox.information(self, "Dashboard", "Run analysis first to generate dashboard.")
            return

        self._add_chart_tab("Dashboard", self._dashboard_path)

    def _load_chart_tabs(self, chart_paths: dict[str, Path]) -> None:
        self.chart_tabs.clear()

        ordered_keys = [
            "dashboard",
            "forecast_overview",
            "moving_averages",
            "returns_distribution",
            "weekday_pattern",
            "model_comparison",
            "forecast_residuals",
        ]

        for key in ordered_keys:
            path = chart_paths.get(key)
            if path is None or not path.exists():
                continue
            self._add_chart_tab(self._tab_title(key), path)

    def _add_chart_tab(self, title: str, path: Path) -> None:
        view = QWebEngineView()
        view.load(QUrl.fromLocalFile(str(path.resolve())))
        self.chart_tabs.addTab(view, title)

    def _tab_title(self, key: str) -> str:
        mapping = {
            "dashboard": "Dashboard",
            "forecast_overview": "Forecast",
            "moving_averages": "Moving Avg",
            "returns_distribution": "Returns",
            "weekday_pattern": "Weekday",
            "model_comparison": "Model Compare",
            "forecast_residuals": "Residuals",
        }
        return mapping.get(key, key)

    def _load_historical_data(self, csv_path: Path) -> pd.DataFrame:
        tgju_data = fetch_gold_history(slug="geram18", unit="toman")
        if tgju_data is not None and not tgju_data.empty:
            save_price_csv(tgju_data, csv_path)
            return tgju_data

        raise RuntimeError("No data loaded from TGJU. Check internet access or TGJU availability.")

    def _refresh_latest_price(self, history: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        live_row = fetch_gold_data(slug="geram18", unit="toman")
        if live_row is None:
            return history

        adjusted_price, _ = _align_price_scale(
            new_price=float(live_row["price"]),
            history_prices=history["price"] if "price" in history.columns else pd.Series(dtype=float),
        )
        live_row["price"] = adjusted_price

        history_price = pd.to_numeric(history["price"], errors="coerce").dropna()
        if history_price.empty:
            return history

        last_price = float(history_price.iloc[-1])
        change_ratio = abs(adjusted_price - last_price) / max(last_price, 1.0)
        if change_ratio > 0.35:
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


def run() -> None:
    app = QApplication(sys.argv)
    window = GoldAnalysisWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
