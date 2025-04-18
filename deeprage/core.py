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

# Helper function to calculate missing value percentage
def missing_percentage(df, col):
    """Calculate the percentage of missing values for a DataFrame column."""
    return np.round(100 - df[col].count() / len(df) * 100, 1)

# Function to generate summary table for missing data analysis
def generate_missing_summary(df_train, df_test, target):
    table = PrettyTable()
    table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %', 'Discrete Ratio (Train)']
    rows = []
    for column in df_train.columns:
        data_type = str(df_train[column].dtype)
        train_missing = missing_percentage(df_train, column)
        test_missing = missing_percentage(df_test, column) if column != target else "NA"
        discrete_ratio = np.round(df_train[column].nunique() / len(df_train), 4)
        rows.append([column, data_type, train_missing, test_missing, discrete_ratio])
    table.add_rows(rows)
    return table

# Function to calculate value counts and percentages
def get_values(df, column, top_n=None, sort=False):
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

# Function for pie chart visualization
def val_pie(df, column, top_n=9, sort=False):
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

# Function for bar chart visualization
def val_bar(df, column, top_n=9, sort=False):
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
        ax.text(x_center, height + max(vc_df['Count']) * 0.02, label,
                ha='center', va='bottom')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.title(f'{column} Count Distribution')
    plt.show()

# Core class for DeepRage functionalities
class RageReport:
    def __init__(self, df_or_path):
        self.df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path

    def clean(self):
        num_cols = self.df.select_dtypes(include='number').columns
        cat_cols = self.df.select_dtypes(include='object').columns
        self.df[num_cols] = SimpleImputer(strategy='median').fit_transform(self.df[num_cols])
        self.df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(self.df[cat_cols])
        return self

    def profile(self, output="report.html"):
        profile = ProfileReport(self.df, title="DeepRage EDA", minimal=True)
        profile.to_file(output)
        return output

    def suggest_features(self):
        for col in self.df.select_dtypes(include='datetime'):
            self.df[f"{col}_sin"] = np.sin(2 * np.pi * self.df[col].dt.dayofyear / 365)
            self.df[f"{col}_cos"] = np.cos(2 * np.pi * self.df[col].dt.dayofyear / 365)
        return [c for c in self.df.columns if c.endswith(("_sin","_cos"))]

    def propose_model(self, target):
        X = self.df.drop(columns=[target])
        y = self.df[target]
        problem = 'classification' if y.nunique() <= 10 and y.dtype in ['int64','object'] else 'regression'
        if problem == 'regression':
            pipe = Pipeline([('scale', StandardScaler()), ('model', Ridge())])
            metric = 'RMSE'
        else:
            pipe = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression())])
            metric = 'ROC-AUC'
        pipe.fit(X, y)
        score = pipe.score(X, y)
        return {"type": problem, "model": pipe.named_steps['model'].__class__.__name__, metric: round(score, 3)}

    def missing_summary(self, df_test, target):
        return generate_missing_summary(self.df, df_test, target)

    def get_values(self, *args, **kwargs):
        return get_values(self.df, *args, **kwargs)

    def val_pie(self, *args, **kwargs):
        return val_pie(self.df, *args, **kwargs)

    def val_bar(self, *args, **kwargs):
        return val_bar(self.df, *args, **kwargs)
