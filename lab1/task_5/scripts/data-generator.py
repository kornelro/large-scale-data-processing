"""Script for generation of artificial datasets."""
import argparse
import pickle
from typing import List, Tuple

import numpy as np


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-samples',
        required=True,
        help='Number of samples to generate',
        type=int,
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Name of directory to save generated data',
        type=str,
    )

    return parser.parse_args()


def generate_data(num_samples: int) -> Tuple[List[float], List[float]]:
    """Generated X, y with given number of data samples."""
    X = np.random.rand(num_samples) * 10
    y = (1.5 * X + 1) + np.random.normal(loc=0, scale=3, size=num_samples)

    return (X, y)


def main() -> None:
    """Runs script."""
    args = get_args()
    data = generate_data(args.num_samples)
    save_dir = ''.join(
        [args.out_dir, '/data', str(args.num_samples), '.pickle']
    )
    pickle.dump(data, open(save_dir, 'wb'))


if __name__ == '__main__':
    main()
