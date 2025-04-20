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
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split
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
    sns.set_style("whitegrid", {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "lightgrey"
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(df[x_col], df[y_col], color="black")
    ax.grid(True)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

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

    def propose_model(self, target, cv: int = 5, include_shap: bool = False):
        """
        Evaluate a suite of models with preprocessing, cross-validation, and optional SHAP.

        Parameters:
        - target: name of the target column
        - cv: number of CV folds
        - include_shap: if True, computes and plots SHAP summary for best model
        """
        # Split features/target
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # Detect classification vs regression
        is_classif = (y.dtype == 'object' or y.nunique() <= 10)

        # Encode categorical target if needed
        if is_classif:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Identify numeric & categorical features
        num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Build preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_feats),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
            ], remainder='drop'
        )

        # Define candidate pipelines
        candidates = {
            'Ridge': Pipeline([('pre', preprocessor), ('model', Ridge())]) if not is_classif else None,
            'RandomForest': Pipeline([
                ('pre', preprocessor),
                ('model', RandomForestClassifier() if is_classif else RandomForestRegressor())
            ]),
            'XGBoost': Pipeline([
                ('pre', preprocessor),
                ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss') if is_classif else XGBRegressor())
            ])
        }
        # Drop invalid candidate
        candidates = {k: v for k, v in candidates.items() if v is not None}

        # Choose scoring
        scoring = 'roc_auc' if is_classif else 'neg_root_mean_squared_error'
        metric_name = 'ROC-AUC' if is_classif else 'RMSE'

        # Cross-validate each
        results = {}
        for name, pipe in candidates.items():
            cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            score = np.mean(cv_res['test_score'])
            if not is_classif:
                score = -score
            results[name] = round(score, 4)

        # Build result table
        table = PrettyTable()
        table.field_names = ['Model', metric_name]
        for model_name, sc in results.items():
            table.add_row([model_name, sc])

        # Fit the best model
        best_model_name = max(results, key=lambda k: results[k] if is_classif else -results[k])
        best_pipe = candidates[best_model_name]
        best_pipe.fit(X, y)

        # Optional SHAP analysis
        if include_shap:
            explainer = shap.Explainer(best_pipe.named_steps['model'],
                                       preprocessor.transform(X))
            shap_values = explainer(preprocessor.transform(X))
            shap.summary_plot(shap_values, features=preprocessor.transform(X))

        return table

    def missing_summary(self, *args):
        """Flexible missing‑data summary."""
        # existing implementation...
        ...

    def ts_plot(self, x_col, y_col, title=None):
        """Instance wrapper for ts_plot helper."""
        ts_plot(self.df, x_col, y_col, title)
