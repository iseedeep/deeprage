import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import shap

# ─── Helper Functions ─────────────────────────────────────────────────────────

def missing_percentage(df, col):
    """Calculate the percentage of missing values for a DataFrame column."""
    return np.round(100 - df[col].count() / len(df) * 100, 1)


def generate_missing_summary(df_train, df_test, target):
    """Generate a PrettyTable summarizing missing % and discrete ratio."""
    table = PrettyTable()
    table.field_names = [
        'Feature',
        'Data Type',
        'Train Missing %',
        'Test Missing %',
        'Discrete Ratio (Train)'
    ]
    rows = []
    for column in df_train.columns:
        data_type = str(df_train[column].dtype)
        train_missing = missing_percentage(df_train, column)
        test_missing = (
            missing_percentage(df_test, column)
            if (df_test is not None and column != target)
            else "NA"
        )
        discrete_ratio = np.round(df_train[column].nunique() / len(df_train), 4)
        rows.append([column, data_type, train_missing, test_missing, discrete_ratio])
    table.add_rows(rows)
    return table


def get_values(df, column, top_n=None, sort=False):
    """
    Return a DataFrame with counts and percentages for each unique value in column.
    """
    vc_df = (
        df.groupby(column)
        .size()
        .reset_index(name='Count')
        .assign(Percentage=lambda x: x['Count'] / x['Count'].sum() * 100)
    )
    if top_n is not None:
        vc_df = vc_df.sort_values(by='Count', ascending=False).head(top_n)
    if sort:
        vc_df = vc_df.sort_values(by=column)
    return vc_df


def val_pie(df, column, top_n=9, sort=False):
    """Plot a black‑themed pie chart of value counts for a categorical column."""
    vc_df = get_values(df, column, top_n, sort)
    colors = plt.cm.Greys(np.linspace(0.9, 0.3, len(vc_df)))
    vc_df.set_index(column).plot.pie(
        figsize=(5, 5),
        y='Count',
        ylabel='',
        legend=False,
        colors=colors
    )
    plt.title(f'{column} Distribution')
    plt.show()


def val_bar(df, column, top_n=9, sort=False):
    """Plot a black‑themed bar chart of value counts with annotations."""
    vc_df = get_values(df, column, top_n, sort)
    vc_df = vc_df.sort_values('Count', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=vc_df,
        x=column,
        y='Count',
        palette='Greys'
    )
    for patch, (_, row) in zip(ax.patches, vc_df.iterrows()):
        x_center = patch.get_x() + patch.get_width() / 2
        height = patch.get_height()
        label = f"{row['Count']:.0f} ({row['Percentage']:.2f}%)"
        ax.text(
            x_center,
            height + max(vc_df['Count']) * 0.02,
            label,
            ha='center',
            va='bottom'
        )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.title(f'{column} Count Distribution')
    plt.show()


def ts_plot(df, x_col, y_col, title=None):
    """Plot a black‑themed time series for a datetime x_col vs numeric y_col."""
    sns.set_style('whitegrid')
    plt.rc(
        'figure',
        autolayout=True,
        figsize=(12, 6),
        titlesize=18,
        titleweight='bold'
    )
    plt.rc(
        'axes',
        labelweight='bold',
        labelsize='large',
        titlesize=16,
        titleweight='bold',
        titlepad=10
    )

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col], color='black')
    ax.set_title(title or f"{y_col} over {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.show()


# ─── RageReport Class ────────────────────────────────────────────────────────

class RageReport:
    def __init__(self, df_or_path):
        """Initialize with a DataFrame or path to CSV."""
        self.df = (
            pd.read_csv(df_or_path)
            if isinstance(df_or_path, str)
            else df_or_path.copy()
        )

    def clean(self):
        """Impute missing values: median for numeric, mode for categorical."""
        num_cols = self.df.select_dtypes(include='number').columns
        cat_cols = self.df.select_dtypes(include='object').columns
        num_imp = SimpleImputer(strategy='median')
        cat_imp = SimpleImputer(strategy='most_frequent')
        self.df[num_cols] = num_imp.fit_transform(self.df[num_cols])
        self.df[cat_cols] = cat_imp.fit_transform(self.df[cat_cols])
        return self

    def profile(self, output="report.html"):
        """Generate a minimal EDA report via ydata‑profiling."""
        profile = ProfileReport(self.df, title="DeepRage Profile", minimal=True)
        profile.to_file(output)
        return output

    def suggest_features(self):
        """Auto‑encode datetime columns into cyclical sine/cosine features."""
        for col in self.df.select_dtypes(include='datetime64[ns]'):
            self.df[f"{col}_sin"] = np.sin(
                2 * np.pi * self.df[col].dt.dayofyear / 365
            )
            self.df[f"{col}_cos"] = np.cos(
                2 * np.pi * self.df[col].dt.dayofyear / 365
            )
        return [c for c in self.df.columns if c.endswith(("_sin", "_cos"))]

    def propose_model(self, target):
        """
        Fit a baseline model:
        - Regression (Ridge) if target is continuous
        - Classification (LogisticRegression) if target is categorical/integer
        Returns a dict with model type, name, and score.
        """
        X = self.df.drop(columns=[target])
        y = self.df[target]
        is_classif = (y.dtype == 'int' or y.dtype == 'object') and y.nunique() <= 10
        if is_classif:
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('model', LogisticRegression())
            ])
            metric = 'ROC‑AUC'
        else:
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('model', Ridge())
            ])
            metric = 'RMSE'
        pipe.fit(X, y)
        score = pipe.score(X, y)
        return {
            "type": "classification" if is_classif else "regression",
            "model": pipe.named_steps['model'].__class__.__name__,
            metric: round(score, 4)
        }

    def missing_summary(self, *args):
        """
        Flexible missing‑data summary:
        - rr.missing_summary('TargetCol')           # train only
        - rr.missing_summary(df_test, 'TargetCol')  # compare train vs. test
        """
        if len(args) == 1:
            df_test = None
            target = args[0]
        elif len(args) == 2:
            df_test, target = args
        else:
            raise TypeError(
                "missing_summary expects either (target) or (df_test, target)"
            )
        # Build table
        table = PrettyTable()
        table.field_names = [
            'Feature',
            'Data Type',
            'Train Missing %',
            'Test Missing %',
            'Discrete Ratio (Train)'
        ]
        rows = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            train_miss = missing_percentage(self.df, col)
            test_miss = (
                missing_percentage(df_test, col)
                if df_test is not None and col != target
                else "NA"
            )
            disc_ratio = np.round(self.df[col].nunique() / len(self.df), 4)
            rows.append([col, dtype, train_miss, test_miss, disc_ratio])
        table.add_rows(rows)
        return table

    def ts_plot(self, x_col, y_col, title=None):
        """Instance wrapper for ts_plot helper."""
        ts_plot(self.df, x_col, y_col, title)
