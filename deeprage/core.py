# core.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from prettytable import PrettyTable
from ydata_profiling import ProfileReport

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from xgboost import XGBClassifier, XGBRegressor
import shap

# ─── Helper Functions ─────────────────────────────────────────────────────────

def missing_percentage(df, col):
    """Calculate the percentage of missing values for a DataFrame column."""
    return np.round(100 - df[col].count() / len(df) * 100, 1)

def generate_missing_summary(df_train, df_test, target):
    """Generate a PrettyTable summarizing missing % and discrete ratio."""
    table = PrettyTable()
    table.field_names = [
        'Feature', 'Data Type',
        'Train Missing %', 'Test Missing %',
        'Discrete Ratio (Train)'
    ]
    rows = []
    for col in df_train.columns:
        dtype = str(df_train[col].dtype)
        train_miss = missing_percentage(df_train, col)
        test_miss = (
            missing_percentage(df_test, col)
            if (df_test is not None and col != target)
            else "NA"
        )
        disc_ratio = np.round(df_train[col].nunique() / len(df_train), 4)
        rows.append([col, dtype, train_miss, test_miss, disc_ratio])
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
        vc_df = vc_df.sort_values('Count', ascending=False).head(top_n)
    if sort:
        vc_df = vc_df.sort_values(column)
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
    vc_df = get_values(df, column, top_n, sort).sort_values('Count', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=vc_df, x=column, y='Count', palette='Greys')
    for patch, (_, row) in zip(ax.patches, vc_df.iterrows()):
        x_c = patch.get_x() + patch.get_width() / 2
        h = patch.get_height()
        lbl = f"{row['Count']:.0f} ({row['Percentage']:.2f}%)"
        ax.text(x_c, h + vc_df['Count'].max() * 0.02, lbl, ha='center', va='bottom')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.title(f'{column} Count Distribution')
    plt.show()

def ts_plot(df, x_col, y_col, title=None):
    """Standalone time‑series plot helper."""
    sns.set_style("whitegrid", {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "lightgrey"
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[x_col], df[y_col], color="black")
    ax.grid(True)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(title or f"{y_col} → {x_col}", weight="bold")
    ax.set_xlabel(x_col, weight="bold")
    ax.set_ylabel(y_col, weight="bold")
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
        """Impute missing: median for numeric (if any), mode for categorical."""
        # coerce numeric‑looking strings
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

        num_cols = self.df.select_dtypes(include='number').columns
        cat_cols = self.df.select_dtypes(include='object').columns

        if len(num_cols) > 0:
            self.df[num_cols] = SimpleImputer(strategy='median') \
                .fit_transform(self.df[num_cols])
        if len(cat_cols) > 0:
            self.df[cat_cols] = SimpleImputer(strategy='most_frequent') \
                .fit_transform(self.df[cat_cols])

        # strip stray whitespace in column names
        self.df.columns = self.df.columns.str.strip()
        return self

    def profile(self, output="report.html"):
        """Generate a minimal EDA report via ydata‑profiling."""
        profile = ProfileReport(self.df, title="DeepRage Profile", minimal=True)
        profile.to_file(output)
        return output

    def suggest_features(self):
        """Auto‑encode datetime columns into sine/cosine features."""
        for col in self.df.select_dtypes(include='datetime64[ns]'):
            self.df[f"{col}_sin"] = np.sin(2 * np.pi * self.df[col].dt.dayofyear / 365)
            self.df[f"{col}_cos"] = np.cos(2 * np.pi * self.df[col].dt.dayofyear / 365)
        return [c for c in self.df.columns if c.endswith(("_sin", "_cos"))]

    def propose_model(self, target, cv=5, include_shap=False):
        """
        Try multiple models with CV and return a PrettyTable of scores.
        Single print only when you do: print(tbl) = rr.propose_model(...)
        """
        # Prepare X, y
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # Detect classification vs regression
        is_classif = (y.dtype == 'object' or y.nunique() <= 10)
        if is_classif:
            # Label‑encode y so classifiers see ints
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Preprocessor
        num_feats = X.select_dtypes(include='number').columns.tolist()
        cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        # Choose metrics, models, and CV splitter
        if is_classif:
            n_classes = len(np.unique(y))
            if n_classes == 2:
                scoring, metric_name = 'roc_auc', 'ROC‑AUC'
                candidates = [
                    ('LogisticRegression', LogisticRegression(max_iter=1000)),
                    ('RandomForest',      RandomForestClassifier()),
                    ('XGBoost',           XGBClassifier(use_label_encoder=False,
                                                        objective='binary:logistic',
                                                        eval_metric='logloss'))
                ]
            else:
                scoring, metric_name = 'accuracy', 'Accuracy'
                # skip XGBoost for multiclass to avoid missing‐class errors
                candidates = [
                    ('LogisticRegression', LogisticRegression(max_iter=1000)),
                    ('RandomForest',      RandomForestClassifier())
                ]
            # stratify only when it’s truly classification
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            scoring, metric_name = 'neg_root_mean_squared_error', 'RMSE'
            candidates = [
                ('Ridge',            Ridge()),
                ('RandomForest',     RandomForestRegressor()),
                ('XGBoost',          XGBRegressor())
            ]
            cv_split = cv

        # Evaluate with CV (errors map to nan)
        table = PrettyTable()
        table.field_names = ['Model', metric_name]
        best_score, best_pipe = -np.inf, None

        for name, mdl in candidates:
            pipe = Pipeline([('pre', pre), ('model', mdl)])
            # default error_score=np.nan will set failed folds to nan
            scores = cross_val_score(pipe, X, y, cv=cv_split, scoring=scoring)
            # average while ignoring nan
            mean_score = (-np.nanmean(scores)
                          if scoring.startswith('neg_')
                          else np.nanmean(scores))
            table.add_row([name, round(mean_score, 4)])
            if mean_score > best_score:
                best_score, best_pipe = mean_score, pipe

        # Optional SHAP (binary only)
        if include_shap and is_classif and metric_name == 'ROC‑AUC':
            best_pipe.fit(X, y)
            X_t = best_pipe.named_steps['pre'].transform(X)
            explainer = shap.Explainer(best_pipe.named_steps['model'], X_t)
            shap_values = explainer(X_t)
            shap.summary_plot(shap_values, features=X, feature_names=X.columns)

        return table

    def missing_summary(self, *args):
        """
        Usage:
          .missing_summary('TargetCol')             # train-only
          .missing_summary(df_test, 'TargetCol')    # train vs. test
        """
        if len(args) == 1:
            return generate_missing_summary(self.df, None, args[0])
        else:
            return generate_missing_summary(self.df, args[0], args[1])

    def ts_plot(self, x_col, y_col, title=None):
        """Instance wrapper for ts_plot helper."""
        ts_plot(self.df, x_col, y_col, title)
