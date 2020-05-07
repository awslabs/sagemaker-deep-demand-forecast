from pathlib import Path

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, load_datasets


def load_multivariate_datasets(path: Path) -> TrainDatasets:
    ds = load_datasets(path / "metadata", path / "train", path / "test")
    target_dim = ds.metadata.feat_static_cat[0].cardinality
    grouper_train = MultivariateGrouper(max_target_dim=target_dim)
    grouper_test = MultivariateGrouper(max_target_dim=target_dim)
    return TrainDatasets(
        metadata=ds.metadata, train=grouper_train(ds.train), test=grouper_test(ds.test),
    )
