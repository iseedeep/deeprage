![](https://www.dropbox.com/scl/fi/zeoyi3ersxj567edjcyh7/Gray-Red-Bold-History-YouTube-Thumbnail.png?rlkey=ix54ns5omxx6yokv4568x0gbg&st=9rrst97j&raw=1)

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

**Command‑Line Interface**
```bash
# Generate HTML EDA report
deeprage profile data.csv

# Missing‑data summary
deeprage missing_summary train.csv [test.csv] TargetColumn

# Fit baseline model
deeprage model data.csv TargetColumn --cv 5 --shap

# Pie & bar charts
deeprage pie data.csv CategoryColumn --top_n 5 --sort
deeprage bar data.csv CategoryColumn --top_n 5 --sort

# Time‑series plot
deeprage ts data.csv DateColumn ValueColumn --title "Trend"

# **Numeric histogram** (counts/KDE or frequency/KDE)
deeprage hist data.csv NumericColumn --bins 30 [--kde] [--freq]
```

### Jupyter Notebook Magics
```python
%load_ext deeprage.notebook

%deeprage_profile data.csv
%deeprage_pie    data.csv CategoryColumn
%deeprage_bar    data.csv CategoryColumn
%deeprage_ts     data.csv DateColumn ValueColumn "Trend"
%deeprage_missing train.csv test.csv TargetColumn
%deeprage_hist   data.csv NumericColumn --bins=30 --kde --freq
```

### Python API

```python
from deeprage.core import (
    RageReport, get_values, val_pie, val_bar, ts_plot, val_hist
)
import pandas as pd

# 1. Load & clean
df = pd.read_csv('data.csv')
rr = RageReport(df).clean()

# 2. Missing‑data summary
print(rr.missing_summary('TargetColumn'))

# 3. Modeling w/ CV=5 + SHAP
rr.propose_model('TargetColumn', cv=5, include_shap=True)

# 4. Categorical distributions
val_pie(df, 'CategoryColumn', top_n=5, sort=True)
val_bar(df, 'CategoryColumn', top_n=5, sort=True)

# 5. Time‑series plotting
ts_plot(df, 'DateColumn', 'ValueColumn', title='Trend')
rr.ts_plot('DateColumn', 'ValueColumn', title='Trend')

# 6. **Numeric histogram**
#    – raw counts + KDE
val_hist(df, 'NumericColumn', bins=30, kde=True,  freq=False)
#    – percent frequencies (no KDE)
val_hist(df, 'NumericColumn', bins=20, kde=False, freq=True)
```

### FastAPI Endpoint
```bash
uvicorn deeprage.api:app --reload
```  
**POST** `/ask` JSON payload as above.

---

> **R·A·G·E** – *Ready your data, Analyze insights, Graphical plots, Explore freely.*


