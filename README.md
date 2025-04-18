# DeepRage

**DeepRage** – Reports · Analytics · Graphical · Explore  
An all‑in‑one Python toolkit for EDA, profiling, visualization, and natural‑language exploration of your datasets.

## Features

- **Reports**: Full HTML EDA report via ydata‑profiling (`deeprage profile`).
- **Analytics**:
  - Missing‑data diagnostics with train/test comparison (`deeprage missing_summary`).
  - Baseline modeling (Ridge/Logistic Regression) with performance metrics (`deeprage model`).
  - Automatic feature suggestions for datetime columns (`RageReport.suggest_features()`).
- **Graphical**:
  - Black‑themed pie charts (`deeprage pie`) and bar charts (`deeprage bar`) with counts & percentages.
- **Explore**:
  - Jupyter magics for inline EDA (`%deeprage_profile`, `%deeprage_pie`, `%deeprage_bar`, `%deeprage_missing`).
  - FastAPI `/ask` endpoint for natural‑language queries (data preview, SHAP feature importance).

## Installation

```bash
# Clone the repo
git clone https://github.com/iseedeep/deeprage.git
cd deeprage

# (Conda)
conda env create -f environment.yml
conda activate deeprage

# or venv
git checkout main
py -3.11 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1

# Editable install
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Generate HTML EDA report
deeprage profile data.csv

# Missing-data summary (train vs. test)
deeprage missing_summary train.csv test.csv TargetColumn

# Fit baseline model and show metrics
deeprage model data.csv TargetColumn

# Pie chart of top N categories
deeprage pie data.csv CategoryColumn --top_n 5 --sort

# Bar chart of top N categories
deeprage bar data.csv CategoryColumn --top_n 5 --sort
```

### Jupyter Notebook Magics

```python
%load_ext deeprage.notebook

# Inline EDA profile
%deeprage_profile data.csv

# Inline pie chart
%deeprage_pie data.csv CategoryColumn

# Inline bar chart
%deeprage_bar data.csv CategoryColumn

# Inline missing-data summary
%deeprage_missing train.csv test.csv TargetColumn
```

### Python API

```python
from deeprage.core import RageReport, val_bar, val_pie
import pandas as pd

# Load data
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

# Missing-data summary
rr = RageReport(df_train).clean()
table = rr.missing_summary(df_test, target='TargetColumn')
print(table)

# Visualize
df = pd.read_csv('data.csv')
val_pie(df, 'CategoryColumn', top_n=5, sort=True)
val_bar(df, 'CategoryColumn', top_n=5, sort=True)
```

### FastAPI Endpoint

```bash
uvicorn deeprage.api:app --reload
```

Send a POST to `/ask` with JSON:

```json
{
  "dataset_path": "data.csv",
  "question": "What are the top features?"
}
```

---

**R·A·G·E** – Ready, Analytic, Graphical, Exploratory; unleash your data insights!

