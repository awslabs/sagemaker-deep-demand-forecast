import numpy as np

from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.artificial import constant_dataset

from preprocess import MaxNormalize


def test_max_normalize():
    info, train_ds, test_ds = constant_dataset()
    datasets = TrainDatasets(info.metadata, train_ds, test_ds)
    normalize = MaxNormalize(datasets).apply()
    assert normalize.datasets.metadata == datasets.metadata
    for i, train_data in enumerate(normalize.datasets.train):
        train = train_data["target"]
        if i == 0:
            assert np.all(train == np.zeros(len(train), dtype=np.float32))
        else:
            assert np.all(train == np.ones(len(train), dtype=np.float32))

    assert normalize.datasets.test is not None
    for i, test_data in enumerate(normalize.datasets.test):
        test = test_data["target"]
        if i == 0:
            assert np.all(test == np.zeros(len(test), dtype=np.float32))
        else:
            assert np.all(test == np.ones(len(test), dtype=np.float32))
