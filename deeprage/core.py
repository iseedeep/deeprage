import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from prettytable import PrettyTable
from ydata_profiling import ProfileReport

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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
        'Feature',
        'Data Type',
        'Train Missing %',
        'Test Missing %',
        'Discrete Ratio (Train)'
    ]
    rows = []
    for column in df_train.columns:
        dtype = str(df_train[column].dtype)
        train_missing = missing_percentage(df_train, column)
        test_missing = (
            missing_percentage(df_test, column)
            if (df_test is not None and column != target)
            else "NA"
        )
        disc_ratio = np.round(df_train[column].nunique() / len(df_train), 4)
        rows.append([column, dtype, train_missing, test_missing, disc_ratio])
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
        ax.text(x_center, height + max(vc_df['Count']) * 0.02,
                label, ha='center', va='bottom')
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
        """Init with DataFrame or CSV path."""
        self.df = (
            pd.read_csv(df_or_path)
            if isinstance(df_or_path, str)
            else df_or_path.copy()
        )

    def clean(self):
        """Impute missing: median for numeric (if any), mode for categorical."""
        # coerce any numeric‐looking objects
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

        num_cols = self.df.select_dtypes(include='number').columns
        cat_cols = self.df.select_dtypes(include='object').columns

        if len(num_cols) > 0:
            num_imp = SimpleImputer(strategy='median')
            self.df[num_cols] = num_imp.fit_transform(self.df[num_cols])

        if len(cat_cols) > 0:
            cat_imp = SimpleImputer(strategy='most_frequent')
            self.df[cat_cols] = cat_imp.fit_transform(self.df[cat_cols])

        # strip column names of accidental whitespace
        self.df.columns = self.df.columns.str.strip()
        return self

    def profile(self, output="report.html"):
        """Full HTML EDA via ydata‑profiling."""
        profile = ProfileReport(self.df, title="DeepRage Profile", minimal=True)
        profile.to_file(output)
        return output

    def suggest_features(self):
        """Auto‑encode datetime columns into sine/cosine cyclical features."""
        for col in self.df.select_dtypes(include='datetime64[ns]'):
            self.df[f"{col}_sin"] = np.sin(2 * np.pi * self.df[col].dt.dayofyear / 365)
            self.df[f"{col}_cos"] = np.cos(2 * np.pi * self.df[col].dt.dayofyear / 365)
        return [c for c in self.df.columns if c.endswith(("_sin", "_cos"))]

    def propose_model(self, target, cv=5, include_shap=False):
        """
        Try multiple models with CV and return a PrettyTable of scores.
        For binary classification you can include SHAP explanations.
        """
        # prepare data
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # build preprocessor
        num_feats = X.select_dtypes(include='number').columns.tolist()
        cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ])

        is_classif = (y.dtype == 'object' or y.nunique() <= 10)
        n_classes = y.nunique()

        # choose metric & models
        if is_classif:
            if n_classes == 2:
                scoring = 'roc_auc'
                metric_name = 'ROC‑AUC'
            else:
                scoring = 'accuracy'
                metric_name = 'Accuracy'

            candidates = [
                ('LogisticRegression', LogisticRegression(max_iter=1000)),
                ('RandomForest', RandomForestClassifier()),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ]
        else:
            scoring = 'neg_root_mean_squared_error'
            metric_name = 'RMSE'
            candidates = [
                ('Ridge', Ridge()),
                ('RandomForest', RandomForestRegressor()),
                ('XGBoost', XGBRegressor())
            ]

        # evaluate with CV
        table = PrettyTable()
        table.field_names = ['Model', metric_name]
        best_score, best_pipe = -np.inf, None

        for name, model in candidates:
            pipe = Pipeline([
                ('pre', preprocessor),
                ('model', model)
            ])
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            mean_score = -scores.mean() if scoring.startswith('neg_') else scores.mean()
            table.add_row([name, round(mean_score, 4)])

            if mean_score > best_score:
                best_score, best_pipe = mean_score, pipe

        print(table)

        # optional SHAP for binary
        if include_shap and is_classif and n_classes == 2:
            best_pipe.fit(X, y)
            # explain on transformed features
            X_trans = best_pipe.named_steps['pre'].transform(X)
            explainer = shap.Explainer(best_pipe.named_steps['model'], X_trans)
            shap_values = explainer(X_trans)
            shap.summary_plot(shap_values, features=X, feature_names=X.columns)

        return table

    def missing_summary(self, *args):
        """
        Usage:
          .missing_summary('TargetCol')             # train-only
          .missing_summary(df_test, 'TargetCol')    # train vs. test
        """
        if len(args) == 1:
            df_test, target = None, args[0]
        else:
            df_test, target = args
        return generate_missing_summary(self.df, df_test, target)

    def ts_plot(self, x_col, y_col, title=None):
        """Instance wrapper for ts_plot helper."""
        ts_plot(self.df, x_col, y_col, title)
