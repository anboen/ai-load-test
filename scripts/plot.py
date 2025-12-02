import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import argparse


def main(base_path):
    df = pd.read_csv(f"{base_path}/all_times.csv", sep=';')
    if df is None:
        raise ValueError("No data found in results folder.")
    plot_size = (25, 17)
    if df[~ df['model'].str.contains("gpt-oss")].shape[0] > 0:
        plot_small(df, base_path, plot_size=plot_size)
    if df[df['model'].str.contains("gpt-oss")].shape[0] > 0:
        plot_gpt(df, base_path, plot_size=plot_size)


def plot_small(df: pd.DataFrame, base_path, plot_size=(10, 4)):
    print("Plotting small models")
    fig, axes = plt.subplots(3, 1, figsize=plot_size)
    df_rest = df[~ df['model'].str.contains("gpt-oss")]

    sns.lineplot(data=df_rest,
                 x='position', y='norm_time', hue='gpu', style='model', ax=axes[0])
    sns.lineplot(data=df_rest,
                 x='position', y='time', hue='gpu', style='model', ax=axes[1])
    sns.lineplot(data=df_rest,
                 x='position', y='diff_time', hue='gpu', style='model', ax=axes[2])
    fig.savefig(f"{base_path}/small_models.png")


def plot_gpt(df: pd.DataFrame, base_path, plot_size=(10, 4)):
    print("Plotting gpt models")
    fig, axes = plt.subplots(3, 1, figsize=plot_size)
    df_gpt = df[df['model'].str.contains("gpt-oss")]
    sns.lineplot(data=df_gpt,
                 x='position', y='norm_time', hue='gpu', style='model', ax=axes[0])
    sns.lineplot(data=df_gpt,
                 x='position', y='time', hue='gpu', style='model', ax=axes[1])
    sns.lineplot(data=df_gpt,
                 x='position', y='diff_time', hue='gpu', style='model', ax=axes[2])
    fig.savefig(f"{base_path}/gpt_models.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Load Tests for AI Models")
    parser.add_argument(
        "base_path", help="base path for results folder")

    args = parser.parse_args()
    main(args.base_path)
