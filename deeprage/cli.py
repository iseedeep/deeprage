import click
import pandas as pd
from deeprage.core import RageReport, val_pie, val_bar

@click.group()
def main():
    """DeepRage Command Line Interface"""
    pass

@main.command()
@click.argument('csv')
def profile(csv):
    """Generate and save an EDA report via ydata-profiling"""
    out = RageReport(csv).clean().profile()
    click.echo(f"Report saved to {out}")

@main.command()
@click.argument('csv')
@click.argument('target')
def model(csv, target):
    """Propose and fit a baseline model"""
    result = RageReport(csv).clean().propose_model(target)
    click.echo(result)

@main.command()
@click.argument('csv')
@click.argument('column')
@click.option('--top_n', default=9, help='Number of top values to display')
@click.option('--sort', is_flag=True, help='Sort values by column')
def pie(csv, column, top_n, sort):
    """Generate a pie chart of value counts"""
    df = pd.read_csv(csv)
    val_pie(df, column, top_n=top_n, sort=sort)

@main.command()
@click.argument('csv')
@click.argument('column')
@click.option('--top_n', default=9, help='Number of top values to display')
@click.option('--sort', is_flag=True, help='Sort values by column')
def bar(csv, column, top_n, sort):
    """Generate a bar chart of value counts"""
    df = pd.read_csv(csv)
    val_bar(df, column, top_n=top_n, sort=sort)

@main.command()
@click.argument('train_csv')
@click.argument('test_csv')
@click.argument('target')
def missing_summary(train_csv, test_csv, target):
    """Show missing data summary for train and test sets"""
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    table = RageReport(df_train).missing_summary(df_test, target)
    click.echo(table)

if __name__ == '__main__':
    main()
