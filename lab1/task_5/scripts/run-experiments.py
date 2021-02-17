"""Script for time measurement experiments on linear regression models."""
import argparse
import os
import pickle
from time import time
from typing import List, Tuple, Type

import lr
import matplotlib.pyplot as plt
import pandas as pd


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets-dir',
        required=True,
        help='Name of directory with generated datasets',
        type=str,
    )

    return parser.parse_args()


def run_experiments(
    models: List[Type[lr.base.LinearRegression]],
    datasets: List[Tuple[List[float], List[float]]],
) -> pd.DataFrame:
    """Run linear regression experiments."""
    df_rows = []

    for data in datasets:
        for model in models:

            model = model()

            start_time = time()
            model.fit(data[0], data[1])
            exec_time = time() - start_time

            df_rows.append({
                'model': get_model_name(model),
                'num_samples': len(data[0]),
                'time': exec_time,
                'coef_0': model._coef[0],
                'coef_1': model._coef[1]})

    df = pd.DataFrame(df_rows).sort_values(['model', 'num_samples'])
    # df = df.set_index(['model', 'num_samples'])
    print(df)
    df = pd.pivot_table(df, index='num_samples', values='time', columns='model')

    return df


def get_model_name(model: lr.base.LinearRegression) -> str:
    """Gives string name for given model."""
    name = ""

    if isinstance(model, lr.LinearRegressionNumpy):
        name = 'Numpy'
    elif isinstance(model, lr.LinearRegressionProcess):
        name = 'Process'
    elif isinstance(model, lr.LinearRegressionSequential):
        name = 'Sequential'
    elif isinstance(model, lr.LinearRegressionThreads):
        name = 'Threads'

    return name


def make_plot(results: pd.DataFrame) -> None:
    """Plot experiments results."""
    results.plot(kind='bar')
    plt.ylabel('time')
    plt.tight_layout()
    plt.savefig('results.jpg')


def main() -> None:
    """Runs script."""
    args = get_args()

    models = [
        lr.LinearRegressionNumpy,
        lr.LinearRegressionProcess,
        lr.LinearRegressionSequential,
        lr.LinearRegressionThreads
    ]

    datasets = []
    file_names = os.listdir(args.datasets_dir)
    for file_name in file_names:
        if file_name.endswith('.pickle'):
            file_dir = ''.join([args.datasets_dir, '/', file_name])
            try:
                with open(file_dir, 'rb') as f:
                    data = pickle.load(f)
                datasets.append(data)
            except Exception as e:
                print(''.join(['Cannot read file ', file_name]))
                print(e)

    results = run_experiments(models, datasets)

    make_plot(results)


if __name__ == '__main__':
    main()
