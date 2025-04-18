# DeepRage

**DeepRage** – Reports · Analytics · Graphical · Explore  
An all‑in‑one Python toolkit for EDA, profiling, visualization, time‑series analysis, and natural‑language exploration of your datasets.

## Features

- **Reports**: Full HTML EDA report via ydata‑profiling (`deeprage profile`).
- **Analytics**:
  - Missing‑data diagnostics with optional train/test comparison (`deeprage missing_summary`).
  - Baseline modeling (Ridge/Logistic Regression) with performance metrics (`deeprage model`).
  - Automatic feature suggestions for datetime columns (`RageReport.suggest_features()`).
- **Graphical**:
  - Black‑themed pie charts (`deeprage pie`) and bar charts (`deeprage bar`) with counts & percentages.
  - Black‑themed time‑series plotting (`deeprage ts`) for datetime vs numeric data.
- **Explore**:
  - Jupyter magics for inline EDA (`%deeprage_profile`, `%deeprage_pie`, `%deeprage_bar`, `%deeprage_ts`, `%deeprage_missing`).
  - FastAPI `/ask` endpoint for natural‑language queries (data preview, SHAP feature importance).

## Installation

```bash
# Clone the repo
git clone https://github.com/iseedeep/deeprage.git
cd deeprage

# (Conda)
conda env create -f environment.yml
conda activate deeprage

# or with venv
git checkout main
py -3.11 -m venv venv
# on Windows
.\venv\Scripts\Activate.ps1
# on Linux/Mac
source venv/bin/activate

# Editable install
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Generate HTML EDA report
deeprage profile data.csv

# Missing-data summary (train-only or train vs. test)
deeprage missing_summary train.csv [test.csv] TargetColumn

# Fit baseline model
deeprage model data.csv TargetColumn

# Category pie chart
deeprage pie data.csv CategoryColumn --top_n 5 --sort

# Category bar chart
deeprage bar data.csv CategoryColumn --top_n 5 --sort

# Time-series plot
deeprage ts data.csv DateColumn ValueColumn --title "Trend"
```

### Jupyter Notebook Magics

```python
%load_ext deeprage.notebook

# EDA profile
%deeprage_profile data.csv

# Pie chart
%deeprage_pie data.csv CategoryColumn

# Bar chart
%deeprage_bar data.csv CategoryColumn

# Time-series plot
%deeprage_ts data.csv DateColumn ValueColumn "Trend"

# Missing-data summary
%deeprage_missing train.csv test.csv TargetColumn
```

### Python API

```python
from deeprage.core import (
    RageReport, get_values, val_pie, val_bar, ts_plot
)
import pandas as pd

# Load data
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

# Missing-data summary
rr = RageReport(df_train).clean()
tbl1 = rr.missing_summary('TargetColumn')            # train-only
tbl2 = rr.missing_summary(df_test, 'TargetColumn')  # train vs. test
print(tbl1)
print(tbl2)

# Pie & bar charts
val_pie(df_train, 'CategoryColumn', top_n=5, sort=True)
val_bar(df_train, 'CategoryColumn', top_n=5, sort=True)

# Time-series plot
ts_plot(df_train, 'DateColumn', 'ValueColumn', title='Trend')
# or via instance
rr.ts_plot('DateColumn', 'ValueColumn', title='Trend')
```

### FastAPI Endpoint

```bash
uvicorn deeprage.api:app --reload
```

POST to `/ask` with JSON:

```json
{
  "dataset_path": "data.csv",
  "question": "What are the top features?"
}
```

---

**R·A·G·E** – Ready your data, Analyze insights, Graphical plots, Explore freely.

