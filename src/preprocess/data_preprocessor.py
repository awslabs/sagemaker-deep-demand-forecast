import os
import os.path as osp
from pathlib import Path
import logging

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets

logger = logging.getLogger(__name__)

os.environ["PREPROCESSING_INPUT_DIR"] = "/opt/ml/processing/input"
os.environ["PREPROCESSING_OUTPUT_DIR"] = "/opt/ml/processing/output"

DATASET_NAME = "electricity"
LOG_CONFIG = os.getenv(
    "LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini"
)


def load_dataset(dataset_name: str, path: Path) -> TrainDatasets:
    dataset = get_dataset(dataset_name, path, regenerate=False)
    target_dim = dataset.metadata.feat_static_cat[0].cardinality
    grouper_train = MultivariateGrouper(max_target_dim=target_dim)
    grouper_test = MultivariateGrouper(max_target_dim=target_dim)
    return TrainDatasets(
        metadata=dataset.metadata,
        train=grouper_train(dataset.train),
        test=grouper_test(dataset.test),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa("--data-dir", type=str, default=os.getenv("PREPROCESSING_INPUT_DIR"))
    aa("--output-dir", type=str, default=os.getenv("PREPROCESSING_OUTPUT_DIR"))

    args = parser.parse_args()
    logger.info(f"passed args: {args}")

    dataset = load_dataset(DATASET_NAME, Path(args.data_dir))

    train_target = pd.DataFrame(next(iter(dataset.train))["target"])
    train_target.describe().to_csv(
        osp.join(args.output_dir, "train_target_summary.csv")
    )
    logger.info(f"train target shape: {train_target.shape}")

    test_target = pd.DataFrame(next(iter(dataset.test))["target"])
    test_target.describe().to_csv(osp.join(args.output_dir, "test_target_summary.csv"))
    logger.info(f"test target shape: {test_target.shape}")
