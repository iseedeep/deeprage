import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from prettytable import PrettyTable
from ydata_profiling import ProfileReport

from matplotlib.ticker import PercentFormatter

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

def ts_plot(
    df,
    x_col,
    y_col,
    title=None,
    resample: str | None = None,
    smooth: int | None = None,
    annotate_last: bool = False,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Time-series / timeline plot.

    • Parses mixed date formats (YYYY, YYYY-MM-DD, ISO…)
    • If y_col is numeric: line + marker plot (with optional resample/smooth)
    • If y_col is non-numeric: tidy scatter timeline with labels
    • Optional last-point annotation
    """
    # ── 1) Copy & parse dates ────────────────────────────────────────────────
    df = df.copy()
    raw_dates = df[x_col].astype(str)

    # Generic parse + year-only fallback
    dates = pd.to_datetime(raw_dates, errors='coerce', infer_datetime_format=True)
    year_mask = raw_dates.str.match(r'^\d{4}$')
    if year_mask.any():
        dates.loc[year_mask] = pd.to_datetime(raw_dates[year_mask], format='%Y')

    if dates.isna().any():
        print(f"⚠️  Coerced {dates.isna().sum()} bad dates in '{x_col}' → dropped")
    df[x_col] = dates.dropna()
    df = df.loc[df[x_col].notna()]

    # Set datetime index
    df.set_index(x_col, inplace=True)

    # ── 2) Select & detect series type ───────────────────────────────────────
    # Single-series only
    if isinstance(y_col, (list, tuple)):
        raise ValueError("ts_plot only supports a single y_col; got list/tuple")

    series = df[y_col]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    sns.set_style("whitegrid", {
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "lightgrey"
    })
    fig, ax = plt.subplots(figsize=figsize)

    if is_numeric:
        # ── Numeric path: resample / smooth / line plot ──────────────────────
        data = series.to_frame()

        if resample:
            data = data.resample(resample).mean()
        if smooth and smooth > 1:
            data = data.rolling(window=smooth, min_periods=1, center=True).mean()

        ax.plot(
            data.index, data[y_col],
            marker='o', linewidth=2, markersize=4, color='black'
        )
        ax.set_ylabel("Value", weight="bold")

    else:
        # ── Timeline path: map each unique label to an integer code ──────────
        cats = pd.Categorical(series)
        codes = cats.codes
        ax.scatter(df.index, codes, marker='o', s=50, color='black')
        for x, y in zip(df.index, codes):
            ax.vlines(x, ymin=-0.5, ymax=y, color='grey', alpha=0.3)

        # label the y-axis with your album titles
        ax.set_yticks(range(len(cats.categories)))
        ax.set_yticklabels(cats.categories)
        ax.set_ylabel(y_col, weight="bold")

    # ── 3) Common styling ───────────────────────────────────────────────────
    ax.set_title(title or f"{y_col} over time", weight="bold")
    ax.set_xlabel(x_col, weight="bold")

    # Smart date ticks
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()

    # ── 4) Annotate last point (optional) ──────────────────────────────────
    if annotate_last:
        if is_numeric:
            x0, y0 = data.index[-1], data[y_col].iloc[-1]
        else:
            x0 = df.index[-1]
            y0 = pd.Categorical(series).codes[-1]
        ax.annotate(
            f"{y0:.2f}" if is_numeric else cats.categories[y0],
            xy=(x0, y0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold"
        )

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
