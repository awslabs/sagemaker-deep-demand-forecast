import os
from pathlib import Path

import numpy as np

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset, save_datasets, TrainDatasets


os.environ["PREPROCESSING_INPUT_DIR"] = "/opt/ml/processing/input"
os.environ["PREPROCESSING_OUTPUT_DIR"] = "/opt/ml/processing/output"


class MaxNormalize:
    """
    Following the LSTNet paper to normalize each time series by its maximum observed value.
    Note that it records the max values in training data and uses them to scale the testing data.
    """

    def __init__(self, datasets: TrainDatasets):
        self.datasets = datasets
        self.freq = datasets.metadata.freq

    @staticmethod
    def _max_normalize(data, scale: float = None):
        if scale is None:
            scale = np.max(np.abs(data["target"]))
            scale = 1.0 if not scale else scale
        data["target"] /= scale
        return data, scale

    def apply(self):
        train_scale = map(self._max_normalize, iter(self.datasets.train))
        unzip_train_scale = list(zip(*train_scale))
        train = ListDataset(unzip_train_scale[0], freq=self.freq)
        scales = unzip_train_scale[1]
        test = None
        if self.datasets.test is not None:
            test_scale = zip(iter(self.datasets.test), scales)
            test = ListDataset(
                map(lambda x: self._max_normalize(x[0], x[1])[0], test_scale),
                freq=self.freq,
            )

        self.datasets = TrainDatasets(self.datasets.metadata, train, test)
        return self

    def save_datasets(self, path, overwrite=True):
        return save_datasets(self.datasets, path, overwrite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa("--dataset-name", type=str, help="Name of the supported dataset")
    aa("--data-dir", type=str, default=os.getenv("PREPROCESSING_INPUT_DIR"))
    aa("--output-dir", type=str, default=os.getenv("PREPROCESSING_OUTPUT_DIR"))

    args = parser.parse_args()

    datasets = get_dataset(args.dataset_name, Path(args.data_dir), regenerate=False)
    normalize = MaxNormalize(datasets).apply().save_datasets(args.output_dir)
