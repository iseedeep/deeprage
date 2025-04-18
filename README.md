# DeepRage

**DeepRage** – Reports · Analytics · Graphical · Explore  
An all‑in‑one Python toolkit for EDA, profiling, visualization and quick NLQ over your datasets.

## Features

- **Profile**: Full EDA/HTML report via ydata‑profiling  
- **Analytics**: Baseline models, missing‑data summary, auto feature suggestions  
- **Graphical**: Black‑theme pie & bar charts with counts & percentages  
- **Explore**: Jupyter magics & FastAPI NLQ endpoint  

## Quickstart

```bash
# clone repo
git clone https://github.com/<YOUR_USERNAME>/deeprage.git
cd deeprage

# (via conda)
conda env create -f environment.yml
conda activate deeprage

# or venv + pip
py -3.11 -m venv venv && .\venv\Scripts\Activate.ps1
pip install -e .

# see help
deeprage --help
