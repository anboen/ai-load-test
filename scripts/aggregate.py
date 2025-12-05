from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os


def add_diff_time(df_current):
    diff_temp = np.diff(df_current['time'].to_numpy())
    df_current['diff_time'] = np.concatenate([np.zeros(1), diff_temp])
    return df_current


def add_norm_time(df_current):
    np_time = df_current['time'].to_numpy()
    min_time = np.min(np_time)
    max_time = np.max(np_time)
    norm = (np_time - min_time) / (max_time - min_time)
    df_current['norm_time'] = norm
    return df_current


def add_df(df, file, gpu):
    temp_df = pd.read_csv(file, sep=';')
    temp_df = clean_df(gpu, temp_df)
    temp_df.sort_values(by=['time'], inplace=True)
    temp_df = add_diff_time(temp_df)
    temp_df = add_norm_time(temp_df)
    df = pd.concat([df, temp_df], ignore_index=True)
    return df


def first_df(file, gpu):
    df = pd.read_csv(file, sep=';')
    df = clean_df(gpu, df)
    df.sort_values(by=['time'], inplace=True)
    df = add_diff_time(df)
    df = add_norm_time(df)

    return df


def sort_df(df):
    df.sort_values(by=['time'], inplace=True)
    df['position'] = np.arange(len(df.index))
    return df


def clean_df(gpu, df):
    df['gpu'] = gpu
    df['response'] = None
    df.drop(df.tail(50).index, inplace=True)
    return df


def main(base_path="./results/"):
    file_path = Path(base_path) / "all_times.csv"
    if file_path.exists():
        os.remove(file_path)

    df = None
    for file in Path(base_path).glob("*.csv"):
        gpu = file.name.split('_')[1]
        if df is None:
            df = first_df(file, gpu)
        else:
            df = add_df(df, file, gpu)
    if df is None:
        raise ValueError("No data found in results folder.")
    df.drop(df[df['gpu'] == 'RTX-PRO-6000-WK'].index, inplace=True)

    df.to_csv(file_path, index=False, sep=';')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Load Tests for AI Models")
    parser.add_argument(
        "base_path", help="base path for results folder")

    args = parser.parse_args()
    main(args.base_path)
