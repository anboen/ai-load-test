import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(base_path):
    df = pd.read_csv(f"{base_path}/all_times.csv", sep=';')
    if df is None:
        raise ValueError("No data found in results folder.")
    plot_size = (15, 10)

    if df[~ df['model'].str.contains("gpt-oss")].shape[0] > 0:
        plot_small(df, base_path, plot_size=plot_size)
    if df[df['model'].str.contains("gpt-oss")].shape[0] > 0:
        plot_gpt(df, base_path, plot_size=plot_size)


def _plot(df: pd.DataFrame, axis, y_col: str, hue: str,  title: str):
    sns.lineplot(data=df,
                 x='position',
                 y=y_col,
                 hue=hue,
                 ax=axis,
                 estimator="median")
    axis.set_ylabel(title)
    axis.set_xlabel("Requests")
    if df[y_col].max() >= 20:
        major_ytick = round(df[y_col].max(), -1)
        step = 20
    elif df[y_col].max() >= 2:
        major_ytick = round(df[y_col].max(), 0)
        step = 10
    else:
        major_ytick = round(df[y_col].max(), 1)
        step = 0.5

    print(f"major_ytick {y_col} {df[y_col].max()}: {major_ytick}")
    axis.set_yticks(
        np.arange(0, major_ytick, step=step))
    axis.set_xticks(
        np.arange(0, df['position'].max(), step=10))


def plot_small(df: pd.DataFrame, base_path, plot_size=(10, 4)):
    print("Plotting small models")
    df_rest = df[~ df['model'].str.contains("gpt-oss")]

    fig, axes = plt.subplots(1, 1, figsize=plot_size)
    _plot(df_rest, axes, 'time', 'gpu', 'time (s)')
    fig.savefig(f"{base_path}/rest_gpu.png")

    fig, axes = plt.subplots(1, 1, figsize=plot_size)
    _plot(df_rest, axes, 'time', 'model', 'time (s)')
    fig.savefig(f"{base_path}/rest_model.png")


def plot_gpt(df: pd.DataFrame, base_path, plot_size=(10, 4)):
    print("Plotting gpt models")
    fig, axes = plt.subplots(2, 1, figsize=plot_size)
    df_gpt = df[df['model'].str.contains("gpt-oss")]

    fig, axes = plt.subplots(1, 1, figsize=plot_size)
    _plot(df_gpt, axes, 'time', 'gpu', 'time (s)')
    fig.savefig(f"{base_path}/gpt_gpu.png")

    fig, axes = plt.subplots(1, 1, figsize=plot_size)
    _plot(df_gpt, axes, 'time', 'model', 'time (s)')
    fig.savefig(f"{base_path}/gpt_model.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Load Tests for AI Models")
    parser.add_argument(
        "base_path", help="base path for results folder")

    args = parser.parse_args()
    main(args.base_path)
