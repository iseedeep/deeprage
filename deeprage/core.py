import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from typing import Literal
from itertools import combinations
from prettytable import PrettyTable
from ydata_profiling import ProfileReport
from scipy.stats import pearsonr, chi2_contingency

from matplotlib.ticker import PercentFormatter
from matplotlib.patheffects import withStroke

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv

from xgboost import XGBClassifier, XGBRegressor
from joblib import Memory
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
    
    # Plot pie chart
    ax = vc_df.set_index(column).plot.pie(
        figsize=(6, 6),
        y='Count',
        ylabel='',
        legend=False,
        colors=colors,
        autopct='%1.1f%%',  
        startangle=90, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': '--'},  # Add edge color and styling
    )

    # Customize text labels' appearance
    for text in ax.texts:
        text.set_fontsize(12)  # Increase font size
        text.set_fontweight('bold')  # Make text bold
        text.set_color('white')  # Change text color to white for visibility
        text.set_path_effects([
            plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')  # Add shadow to text
        ])

    # Set title and display the plot
    plt.title(f'{column} Distribution', fontsize=14, fontweight='bold')
    plt.show()

def val_bar(df, column, top_n=9, sort=False):
    """Plot bar chart of value counts with annotations."""
    vc_df = get_values(df, column, top_n, sort).sort_values('Count', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the bar chart
    sns.barplot(data=vc_df, x=column, y='Count', palette='Greys', ax=ax)
    
    # Add annotations with better formatting
    for patch, (_, row) in zip(ax.patches, vc_df.iterrows()):
        x_c = patch.get_x() + patch.get_width() / 2
        h = patch.get_height()
        lbl = f"{row['Count']:.0f} ({row['Percentage']:.2f}%)"
        ax.text(
            x_c, h + vc_df['Count'].max() * 0.02, lbl, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='white',
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')]
        )
    
    # Improve ticks and labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12, fontweight='bold')
    ax.set_yticklabels([f'{int(i)}' for i in ax.get_yticks()], fontsize=12, fontweight='bold')
    
    # Set title and improve grid
    ax.set_title(f'{column} Count Distribution', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def val_hist(df, column, bins=30, kde=False, freq=False):
    """
    Plot a histogram for a numeric column:
      – freq=False → y axis is count
      – freq=True  → y axis is relative frequency (%) 
    Also prints a PrettyTable of summary stats.
    """
    # Prepare data
    series = df[column].dropna()
    
    # Summary stats
    stats = {
        'count':  series.count(),
        'mean':   series.mean(),
        'std':    series.std(),
        'min':    series.min(),
        '25%':    series.quantile(0.25),
        '50%':    series.median(),
        '75%':    series.quantile(0.75),
        'max':    series.max()
    }
    table = PrettyTable(['Statistic', 'Value'])
    for k, v in stats.items():
        table.add_row([k, round(v, 4)])
    print(table)
    
    # Plot setup with enhanced style
    sns.set_style("whitegrid", {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "lightgrey"
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Choose stat mode
    stat = 'probability' if freq else 'count'
    ylabel = 'Frequency (%)' if freq else 'Count'
    
    # Draw histogram (and optional KDE)
    sns.histplot(
        series,
        bins=bins,
        stat=stat,
        kde=kde,
        element='step',
        fill=True,
        color='black',
        edgecolor='grey',
        ax=ax
    )
    
    # If freq mode, format y‑axis as percentages
    if freq:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    # If KDE is enabled, adjust line properties
    if kde:
        for line in ax.lines:
            line.set_color('grey')
            line.set_linewidth(2)
    
    # Add customized labels & layout
    ax.set_title(f"{column} Distribution", weight="bold", fontsize=16)
    ax.set_xlabel(column, weight="bold", fontsize=14)
    ax.set_ylabel(ylabel, weight="bold", fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

def val_top_n(
    df,
    category_col: str,
    value_col: str,
    top_n: int = 5,
    agg_method: str = 'sum',
    horizontal: bool | None = None,
    figsize: tuple[int, int] = (12, 6)
):
    # 1) Aggregate
    if agg_method == 'sum':
        agg = df.groupby(category_col)[value_col].sum()
    elif agg_method == 'mean':
        agg = df.groupby(category_col)[value_col].mean()
    else:
        raise ValueError("agg_method must be 'sum' or 'mean'")
    summary = (
        agg
        .reset_index(name='Value')
        .assign(Percentage=lambda x: x['Value'] / x['Value'].sum() * 100)
        .sort_values('Value', ascending=False)
        .head(top_n)
    )

    # 2) Auto-orient if unspecified
    max_label_len = summary[category_col].str.len().max()
    if horizontal is None:
        horizontal = max_label_len > 10 or top_n > 7

    # 3) Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Greys(np.linspace(0.7, 0.2, len(summary)))

    if horizontal:
        sns.barplot(data=summary, y=category_col, x='Value', palette=colors, ax=ax)
    else:
        sns.barplot(data=summary, x=category_col, y='Value', palette=colors, ax=ax)

    # 4) Improved in-bar annotations
    stroke = withStroke(linewidth=3, foreground='black')
    for bar, (_, row) in zip(ax.patches, summary.iterrows()):
        val = row['Value']
        pct = row['Percentage']
        txt = f"{int(val):,} ({pct:.1f}%)"

        if horizontal:
            # center text inside the bar
            x = bar.get_width() / 2
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                x, y, txt,
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='white',
                path_effects=[stroke]
            )
        else:
            # rotate label and put it inside the bar
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() * 0.5
            ax.text(
                x, y, txt,
                ha='center', va='center',
                rotation=90,
                fontsize=12, fontweight='bold', color='white',
                path_effects=[stroke]
            )

    # 5) Final styling
    if horizontal:
        ax.set_ylabel(category_col, fontsize=14, fontweight='bold')
        ax.set_xlabel(value_col, fontsize=14, fontweight='bold')
        fig.subplots_adjust(left=0.3)
    else:
        ax.set_xlabel(category_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(value_col, fontsize=12, fontweight='bold')
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha='right',
            fontsize=10, fontweight='bold'
        )
        fig.subplots_adjust(bottom=0.3)

    ax.set_title(f"Top {top_n} {category_col} by {value_col}", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary

def val_all_hist(df, bins=30, kde=False, freq=False, n_cols=3):
    """
    Plot histograms for all numeric columns in a grid layout.
    
    Parameters:
      df       – DataFrame
      bins     – number of bins per histogram
      kde      – whether to overlay a KDE curve
      freq     – if True, y‑axis shows percentage; otherwise raw counts
      n_cols   – how many subplots per row
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not num_cols:
        print("No numeric columns to plot.")
        return

    n = len(num_cols)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    axes = axes.flatten()

    for ax, col in zip(axes, num_cols):
        series = df[col].dropna()
        stat = 'probability' if freq else 'count'
        ylabel = 'Frequency (%)' if freq else 'Count'
        
        # Plot histogram with optional KDE
        sns.histplot(
            series,
            bins=bins,
            stat=stat,
            kde=kde,
            element='step',
            fill=True,
            color='black',
            edgecolor='grey',
            ax=ax
        )
        
        # Format y-axis as percentage if needed
        if freq:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        # Customize the appearance of each subplot
        ax.set_title(col, weight="bold", fontsize=14)
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Remove any extra axes if the number of columns is not a perfect multiple
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def val_seasonality(
    df,
    date_col: str,
    value_col: str,
    period: Literal['M','W','Y'] = 'M',
    kind: Literal['bar','line'] = 'bar',
    smooth: int | None = None,
    figsize: tuple[int,int] = (12,6)
):
    """
    Plot seasonality of `value_col` over:
      • 'M' → month (Jan…Dec)
      • 'W' → weekdays (Mon…Sun)
      • 'Y' → calendar years
    Choose 'bar' or 'line' and optional moving average.
    """
    # 1) Prep
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], format='%d-%m-%y', errors='coerce')
    df2 = df2.dropna(subset=[date_col, value_col])
    df2 = df2.set_index(date_col).sort_index()

    # 2) Aggregate
    if period == 'M':
        serie = df2[value_col].resample('M').sum()
        labels = serie.index.month_name().str[:3]
        xlabel = "Month"
    elif period == 'W':
        daily = df2[value_col].resample('D').sum()
        serie = daily.groupby(daily.index.day_name()).sum().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )
        labels = serie.index
        xlabel = "Day of Week"
    else:  # 'Y'
        serie = df2[value_col].resample('Y').sum()
        labels = serie.index.year.astype(str)
        xlabel = "Year"

    if smooth and smooth > 1:
        serie = serie.rolling(window=smooth, center=True, min_periods=1).mean()

    # 3) Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid", {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "lightgrey"
    })

    if kind == 'bar':
        ax.bar(labels, serie.values, color='black', edgecolor='grey')
    else:
        ax.plot(labels, serie.values, marker='o', lw=2, ms=6, color='black')

    # 4) Style & annotate
    ax.set_title(f"{value_col} Seasonality ({'Monthly' if period=='M' else 'Weekly' if period=='W' else 'Annual'})",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    if kind == 'bar':
        for x, y in zip(labels, serie):
            ax.text(x, y + serie.max()*0.02, f"{y:.0f}",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    path_effects=[withStroke(linewidth=3, foreground='white')])

    plt.tight_layout()
    plt.show()

    return serie

def compare_columns(df, figsize=(8, 5), corr_alpha=0.05):
    """
    For every pair of columns in df:
      - Num vs Num   : Pearson r + scatterplot
      - Cat vs Num   : Boxplot of num by category
      - Cat vs Cat   : Crosstab + chi-square + heatmap
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1) Numeric vs Numeric
    for x, y in combinations(num_cols, 2):
        clean = df[[x, y]].dropna()
        if clean.empty: continue
        
        r, p = pearsonr(clean[x], clean[y])
        sig = "✓" if p < corr_alpha else "✗"
        print(f"\n▶ {x} ↔ {y}   Pearson r={r:.2f} (p={p:.3g}) {sig}")
        
        plt.figure(figsize=figsize)
        sns.scatterplot(data=clean, x=x, y=y, color='black', edgecolor='white')
        plt.title(f"{x} vs {y}", fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # 2) Categorical vs Numeric
    for cat, num in [(c, n) for c in cat_cols for n in num_cols]:
        clean = df[[cat, num]].dropna()
        if clean.empty: continue
        
        print(f"\n▶ {cat} → {num}")
        display(clean.groupby(cat)[num].agg(['count','mean','std']).round(2))
        
        plt.figure(figsize=figsize)
        sns.boxplot(data=clean, x=cat, y=num, palette=['#333'], fliersize=3)
        plt.title(f"{num} by {cat}", fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 3) Categorical vs Categorical
    for c1, c2 in combinations(cat_cols, 2):
        clean = df[[c1, c2]].dropna()
        if clean.empty: continue
        
        ct = pd.crosstab(clean[c1], clean[c2])
        chi2, p, _, _ = chi2_contingency(ct)
        print(f"\n▶ {c1} ↔ {c2}   χ² p={p:.3g}")
        display(ct)
        
        # heatmap of proportions
        prop = ct.div(ct.sum(axis=1), axis=0)
        plt.figure(figsize=figsize)
        sns.heatmap(prop, annot=True, fmt=".2f", cmap="Greys", cbar=False)
        plt.title(f"{c1} vs {c2} (%)", fontweight='bold')
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
        - Ridge now uses solver='lsqr' (sparse-friendly).
        - Any fit errors are caught and reported per model.
        - If target is missing, raises a clear ValueError.
        """
        # 1) ensure target exists
        if target not in self.df.columns:
            raise ValueError(
                f"Target column {target!r} not found. Available columns: "
                f"{', '.join(self.df.columns)}"
            )

        # 2) prepare X, y
        X = self.df.drop(columns=[target])
        y = self.df[target]

        # 3) detect problem type
        is_classif = (y.dtype == 'object' or y.nunique() <= 10)
        if is_classif:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # 4) preprocessor
        num_feats = X.select_dtypes(include='number').columns.tolist()
        cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=True), cat_feats)
        ])

        # 5) choose metrics, models, splitter
        if is_classif:
            # binary vs multiclass
            n_classes = len(np.unique(y))
            if n_classes == 2:
                scoring, metric = 'roc_auc', 'ROC-AUC'
                candidates = [
                    ('Logistic', LogisticRegression(max_iter=1000)),
                    ('Forest',   RandomForestClassifier()),
                    ('XGBoost',  XGBClassifier(use_label_encoder=False,
                                                objective='binary:logistic',
                                                eval_metric='logloss'))
                ]
            else:
                scoring, metric = 'accuracy', 'Accuracy'
                candidates = [
                    ('Logistic', LogisticRegression(max_iter=1000)),
                    ('Forest',   RandomForestClassifier())
                ]
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            scoring, metric = 'neg_root_mean_squared_error', 'RMSE'
            candidates = [
                ('Ridge (lsqr)',     Ridge(solver='lsqr')),   # sparse-friendly
                ('Forest',           RandomForestRegressor()),
                ('XGBoost',          XGBRegressor())
            ]
            cv_split = cv

        # 6) evaluate with CV, catching errors
        from prettytable import PrettyTable
        table = PrettyTable(['Model', metric])
        best_score, best_pipe = -np.inf, None
        for name, mdl in candidates:
            try:
                pipe = Pipeline([('pre', pre), ('model', mdl)])
                scores = cross_val_score(pipe, X, y,
                                         cv=cv_split,
                                         scoring=scoring)
                # flip if negative scoring
                mean_score = (-np.nanmean(scores)
                              if scoring.startswith('neg_')
                              else np.nanmean(scores))
                table.add_row([name, round(mean_score, 4)])
                if mean_score > best_score:
                    best_score, best_pipe = mean_score, pipe
            except Exception as e:
                # report and keep going
                table.add_row([name, 'FAIL'])
                print(f"→ {name} failed: {e}")

        # 7) if all failed, warn
        if best_pipe is None:
            print("⚠️ All candidate models failed. Check data, types, or configs.")
            return table

        # 8) optional SHAP (binary ROC-AUC only)
        if include_shap and is_classif and metric == 'ROC-AUC':
            best_pipe.fit(X, y)
            X_t = best_pipe.named_steps['pre'].transform(X)
            explainer = shap.Explainer(best_pipe.named_steps['model'], X_t)
            shap.summary_plot(explainer(X_t), features=X, feature_names=X.columns)

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
    
    def competition_model(
        self,
        df_train,
        df_test,
        target,
        id_col=None,
        cv=5,
        tune=False,
        tune_iter=10,           # tune_iter won’t limit Halving search, it's here for API consistency
        random_state=42
    ):
        """
        Competition-ready pipeline with caching, halving search, and full-core trees:
          1) Clean train/test
          2) CV (+ optional HalvingRandomSearchCV on Forest/XGBoost)
          3) Hold-out evaluation
          4) Writes submission.csv using id from index or column
        """
        # 1) Prepare & clean
        self.df = df_train.copy()
        self.clean()
        train = self.df
        test  = pd.read_csv(df_test) if isinstance(df_test, str) else df_test.copy()

        # 2) Split features/target (id in index)
        X_train = train.drop(columns=[target])
        y_train = train[target]
        X_test  = test.drop(columns=[target], errors='ignore')

        # 3) Detect problem type & encode labels
        is_classif = (y_train.dtype == 'object' or y_train.nunique() <= 10)
        if is_classif:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)

        # 4) Build preprocessor
        num_feats = X_train.select_dtypes(include='number').columns.tolist()
        cat_feats = X_train.select_dtypes(include=['object','category']).columns.tolist()
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
        ])

        # 5) Define models & CV splitter
        if is_classif:
            scoring = 'roc_auc' if len(np.unique(y_train))==2 else 'accuracy'
            candidates = {
                'Logistic': LogisticRegression(max_iter=1000, random_state=random_state),
                'Forest':   RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
                'XGBoost':  XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                          random_state=random_state, n_jobs=-1)
            }
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            scoring = 'neg_root_mean_squared_error'
            candidates = {
                'Ridge':   Ridge(solver='lsqr', random_state=random_state),
                'Forest':  RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                'XGBoost': XGBRegressor(random_state=random_state, n_jobs=-1)
            }
            cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        # 6) Hyperparameter grids for halving search
        tunable = {'Forest','XGBoost'}
        param_distributions = {
            'Forest': {
                'model__n_estimators': [50,100],
                'model__max_depth':    [None,5],
            },
            'XGBoost': {
                'model__n_estimators':   [50,100],
                'model__max_depth':      [3,5],
                'model__learning_rate':  [0.1],
            }
        }

        # 7) Prepare cache for preprocessing
        memory = Memory(location="cache_dir", verbose=0)

        # 8) CV evaluation & hold-out
        table = PrettyTable(['Model','CV Score','Hold-out Score'])
        best_score, best_pipe = -np.inf, None

        for name, model in candidates.items():
            # ensure full-core tree training; pipeline caching
            pipe = Pipeline([('pre', pre), ('model', model)], memory=memory)

            if tune and name in tunable:
                search = HalvingRandomSearchCV(
                    pipe,
                    param_distributions[name],
                    scoring=scoring,
                    cv=cv_split,
                    factor=2,
                    random_state=random_state,
                    verbose=1,
                    n_jobs=1       # CV forks still single-process
                )
                search.fit(X_train, y_train)
                pipe = search.best_estimator_
                cv_score = search.best_score_
            else:
                scores = cross_val_score(
                    pipe, X_train, y_train,
                    cv=cv_split,
                    scoring=scoring,
                    n_jobs=1       # CV forks single-process
                )
                cv_score = np.mean(scores)

            # fit full train & predict test
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            # compute hold-out if true labels exist
            if target in test.columns:
                y_true = test[target]
                if is_classif:
                    hold_score = (roc_auc_score(y_true, pipe.predict_proba(X_test)[:,1])
                                  if scoring=='roc_auc'
                                  else accuracy_score(y_true, preds))
                else:
                    hold_score = mean_squared_error(y_true, preds, squared=False)
            else:
                hold_score = 'N/A'

            table.add_row([
                name,
                round(cv_score,4),
                round(hold_score,4) if isinstance(hold_score,float) else hold_score
            ])

            if isinstance(hold_score,float) and hold_score>best_score:
                best_score, best_pipe = hold_score, pipe

        # 9) Build submission.csv from index or column
        final_preds = best_pipe.predict(X_test)
        if id_col and test.index.name==id_col:
            submission = pd.DataFrame({id_col: test.index, target: final_preds})
        elif id_col and id_col in test.columns:
            submission = test[[id_col]].copy()
            submission[target] = final_preds
        else:
            submission = pd.DataFrame({target: final_preds}, index=test.index)

        submission.to_csv('submission.csv', index=False)
        print("🔥 submission.csv is ready!")
        return table, 'submission.csv'
