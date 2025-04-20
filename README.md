# DeepRage
**Reports · Analytics · Graphical · Explore**

DeepRage is an all‑in‑one Python toolkit for EDA, profiling, visualization, time‑series analysis, and natural‑language exploration of your datasets.

---

## 🚀 Features

### 1. Reports
- **Full HTML EDA report** via ydata‑profiling  
  ```bash
  deeprage profile data.csv
  ```

### 2. Analytics & Modeling
- **Missing‑data diagnostics** (train-only or train vs. test)  
  ```bash
  deeprage missing_summary train.csv [test.csv] TargetColumn
  ```
- **Modeling**: baseline pipelines for regression & classification (Ridge, LogisticRegression, RandomForest, XGBoost) with cross-validation and optional SHAP explainability  
  ```bash
  deeprage model data.csv TargetColumn --cv 5 --shap
  ```
  ```python
  from deeprage.core import RageReport
  rr = RageReport('data.csv').clean()
  rr.propose_model('TargetColumn', cv=5, include_shap=True)
  ```
- **Automatic feature suggestions** for datetime columns  
  ```python
  rr.suggest_features()
  ```

### 3. Graphical
- **Pie & bar charts** with counts & percentages  
  ```bash
  deeprage pie data.csv CategoryColumn --top_n 5 --sort
  deeprage bar data.csv CategoryColumn --top_n 5 --sort
  ```
- **Time‑series plotting** for datetime vs numeric data  
  ```bash
  deeprage ts data.csv DateColumn ValueColumn --title "Trend"
  ```

### 4. Explore
- **Jupyter magics** for inline EDA  
  ```bash
  %load_ext deeprage.notebook
  %deeprage_profile data.csv
  %deeprage_pie data.csv CategoryColumn
  %deeprage_bar data.csv CategoryColumn
  %deeprage_ts data.csv DateColumn ValueColumn "Trend"
  %deeprage_missing train.csv test.csv TargetColumn
  ```
- **FastAPI `/ask` endpoint** for natural‑language queries  
  ```bash
  uvicorn deeprage.api:app --reload
  ```
  **POST** `/ask` with JSON:  
  ```json
  {
    "dataset_path": "data.csv",
    "question": "What are the top features?"
  }
  ```

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/iseedeep/deeprage.git
   cd deeprage
   ```
2. **Using Conda**  
   ```bash
   conda env create -f environment.yml
   conda activate deeprage
   ```
3. **Using venv**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\\venv\\Scripts\\Activate.ps1  # Windows PowerShell
   ```
4. **Editable install**  
   ```bash
   pip install -e .
   ```

---

## 💻 Usage

### Command‑Line Interface
```bash
# Generate HTML EDA report
deeprage profile data.csv

# Missing-data summary
deeprage missing_summary train.csv [test.csv] TargetColumn

# Fit baseline model
deeprage model data.csv TargetColumn --cv 5 --shap

# Pie & bar charts
deeprage pie data.csv CategoryColumn --top_n 5 --sort
deeprage bar data.csv CategoryColumn --top_n 5 --sort

# Time-series plot
deeprage ts data.csv DateColumn ValueColumn --title "Trend"
```

### Jupyter Notebook Magics
```python
%load_ext deeprage.notebook
%deeprage_profile data.csv
%deeprage_pie data.csv CategoryColumn
%deeprage_bar data.csv CategoryColumn
%deeprage_ts data.csv DateColumn ValueColumn "Trend"
%deeprage_missing train.csv test.csv TargetColumn
```

### Python API

```python
from deeprage.core import (
    RageReport, get_values, val_pie, val_bar, ts_plot
)
import pandas as pd

# 1. Load data and clean missing values (median for numeric, mode for categorical)
df_train = pd.read_csv('train.csv')
rr = RageReport(df_train).clean()

# 2. Missing‑data summary: train-only diagnostics (Feature, dtype, missing%, unique ratio)
tbl = rr.missing_summary('TargetColumn')
print(tbl)

# 3. Modeling: fit baseline pipelines (Ridge/Logistic, RF, XGBoost) with CV=5 and display SHAP plot if requested
rr.propose_model('TargetColumn', cv=5, include_shap=True)

# 4. Categorical distributions: pie chart for top 5 categories, sorted by count
val_pie(df_train, 'CategoryColumn', top_n=5, sort=True)

# 5. Categorical distributions: bar chart with counts & percentages
val_bar(df_train, 'CategoryColumn', top_n=5, sort=True)

# 6. Time‑series plotting: standalone function for datetime vs numeric trend
ts_plot(df_train, 'DateColumn', 'ValueColumn', title='Trend')

# 7. Time‑series plotting via instance wrapper (same as ts_plot above)
rr.ts_plot('DateColumn', 'ValueColumn', title='Trend')
```

### FastAPI Endpoint
```bash
uvicorn deeprage.api:app --reload
```  
**POST** `/ask` JSON payload as above.

---

> **R·A·G·E** – *Ready your data, Analyze insights, Graphical plots, Explore freely.*


