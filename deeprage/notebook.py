# deeprage/notebook.py
import pandas as pd
from IPython.core.magic import register_line_magic
from deeprage.core import val_pie, val_bar, ts_plot

@register_line_magic
def deeprage_pie(line):
    """Jupyter magic for pie chart."""
    csv_file, column_name = line.strip().split()
    df = pd.read_csv(csv_file)
    val_pie(df, column_name)

@register_line_magic
def deeprage_bar(line):
    """Jupyter magic for bar chart."""
    csv_file, column_name = line.strip().split()
    df = pd.read_csv(csv_file)
    val_bar(df, column_name)

@register_line_magic
def deeprage_ts(line):
    """
    Usage: %deeprage_ts path/to/data.csv DateCol ValueCol [Title]
    """
    parts = line.strip().split()
    csv, x_col, y_col = parts[:3]
    title = parts[3] if len(parts) > 3 else None
    df = pd.read_csv(csv, parse_dates=[x_col])
    ts_plot(df, x_col, y_col, title)