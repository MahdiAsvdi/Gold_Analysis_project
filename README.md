# Gold Analysis Project

پروژه تحلیل و پیش‌بینی قیمت طلا با داده‌ی مستقیم از TGJU، خروجی CLI/PyQt، و داشبورد HTML.

## ویژگی‌ها
- منبع داده: فقط `TGJU`
- نماد پیش‌فرض: `geram18` (طلای 18 عیار)
- واحد قیمت در کل پروژه: `تومان`
- پیش‌بینی با مدل ترکیبی (Weighted Ensemble) و بک‌تست چندبخشی
- خروجی گزارش متنی + نمودار Matplotlib + چند نمودار HTML تعاملی
- نمایش نمودارهای HTML داخل PyQt (تب‌بندی‌شده)

## منبع داده
دریافت تاریخچه از API رسمی TGJU:
- `https://api.tgju.org/v1/market/indicator/summary-table-data/geram18?lang=fa&order_dir=asc`

دریافت قیمت لحظه‌ای نیز از TGJU انجام می‌شود (با fallback به صفحه پروفایل در صورت نیاز).

## پیش‌نیازها
- Python 3.10+
- Windows/Linux/macOS

نصب وابستگی‌ها:

```bash
pip install -r requirements.txt
```

## اجرای CLI

```bash
python main.py
```

گزینه‌های مهم:
- `--forecast-days`: تعداد روزهای پیش‌بینی (پیش‌فرض: 30)
- `--language fa|finglish`: زبان خروجی گزارش/نمودار
- `--no-plot`: عدم نمایش نمودار Matplotlib
- `--no-html`: عدم تولید فایل‌های HTML
- `--html-dir`: مسیر ذخیره نمودارهای HTML
- `--report-path`: مسیر ذخیره گزارش متنی
- `--plot-path`: مسیر ذخیره نمودار PNG
- `--refresh-live-price`: افزودن آخرین قیمت آنلاین TGJU به دیتاست

نمونه:

```bash
python main.py --language fa --forecast-days 20
python main.py --no-plot --language finglish
```

## اجرای اپ PyQt

```bash
python pyqt_app.py
```

قابلیت‌ها:
- اجرای تحلیل با یک کلیک
- نمایش گزارش در پنل چپ
- نمایش نمودارهای HTML در تب‌های داخلی
- امکان Refresh Live Price

## خروجی‌ها
پیش‌فرض خروجی‌ها در مسیر `data/`:
- `gold_prices.csv` داده‌ی نرمال‌شده (تومان)
- `analysis_report.txt` گزارش تحلیلی
- `forecast_plot.png` نمودار Matplotlib
- `html_charts/` نمودارهای HTML:
  - `01_forecast_overview.html`
  - `02_moving_averages.html`
  - `03_returns_distribution.html`
  - `04_weekday_pattern.html`
  - `05_model_comparison.html`
  - `06_forecast_residuals.html`
  - `dashboard.html`

## رفع Warningهای Matplotlib
در پروژه موارد زیر مدیریت شده‌اند:
- Warning مربوط به `labels` در `boxplot()` (تغییر به `tick_labels`)
- Warningهای مربوط به Glyph/Arabic با fallback فونت و fallback به finglish

برای کدهای نوت‌بوک خودت از wrapper سازگار استفاده کن:

```python
import matplotlib.pyplot as plt
from src.utils import compatible_boxplot

fig, ax = plt.subplots()
compatible_boxplot(ax, [profits], labels=["Profit"], patch_artist=True)
plt.tight_layout()
```

## ساختار پروژه
- `main.py`: اجرای خط فرمان
- `pyqt_app.py`: رابط گرافیکی PyQt
- `src/scraper.py`: دریافت داده از TGJU
- `src/model.py`: پیش‌بینی و بک‌تست
- `src/utils.py`: گزارش، رسم، ابزارهای کمکی
- `src/html_charts.py`: تولید نمودارهای HTML

## نکات
- اتصال اینترنت برای دریافت داده از TGJU الزامی است.
- این پروژه آموزشی است و خروجی آن توصیه سرمایه‌گذاری محسوب نمی‌شود.
