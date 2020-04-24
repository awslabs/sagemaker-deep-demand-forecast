from pathlib import Path

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets


def load_dataset(dataset_name: str, path: Path, regenerate=False) -> TrainDatasets:
    dataset = get_dataset(dataset_name, path, regenerate)
    target_dim = dataset.metadata.feat_static_cat[0].cardinality
    grouper_train = MultivariateGrouper(max_target_dim=target_dim)
    grouper_test = MultivariateGrouper(max_target_dim=target_dim)
    return TrainDatasets(
        metadata=dataset.metadata,
        train=grouper_train(dataset.train),
        test=grouper_test(dataset.test),
    )
