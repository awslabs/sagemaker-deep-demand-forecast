from typing import Iterator, Dict, Tuple, Any
import os
import os.path as osp
from pathlib import Path
import json
import pickle

import mxnet as mx

from gluonts.model.lstnet import LSTNetEstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import TrainDatasets
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Forecast

from data import load_dataset
from metrics import rse
from utils import get_logger, evaluate


LOG_CONFIG = os.getenv(
    "LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini"
)

logger = get_logger(LOG_CONFIG)


def train(
    dataset: TrainDatasets,
    output_dir: str,
    model_dir: str,
    epochs: int,
    context_length: int,
    prediction_length: int,
    skip_size: int,
    ar_window: int,
    rnn_num_layers: int,
    skip_rnn_num_layers: int,
    channels: int,
    seed: int,
) -> Predictor:
    mx.random.seed(seed)
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    logger.info(f"Using the context: {ctx}")
    trainer_hyperparameters = {
        "ctx": ctx,
        "epochs": epochs,
        "hybridize": True,
        "patience": 10,
        "learning_rate_decay_factor": 0.5,
        "batch_size": 64,
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
    }
    model_hyperparameters = {
        "freq": dataset.metadata.freq,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "skip_size": skip_size,
        "ar_window": ar_window,
        "num_series": dataset.metadata.feat_static_cat[0].cardinality,
        "rnn_num_layers": rnn_num_layers,
        "skip_rnn_num_layers": skip_rnn_num_layers,
        "channels": channels,
        "output_activation": None,
        "trainer": Trainer(**trainer_hyperparameters),
    }
    estimator = LSTNetEstimator(**model_hyperparameters)
    predictor = estimator.train(dataset.train)
    logger.info("Training is done!")
    return predictor


def save(model: Predictor, model_dir: str) -> None:
    model.serialize(Path(model_dir))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa(
        "--dataset_path",
        type=str,
        default=os.environ["SM_CHANNELS"],
        help="path to the dataset",
    )
    aa(
        "--output_dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="output directory",
    )
    aa(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="model directory",
    )
    aa("--dataset_name", type=str, help="name of an available dataset in GluonTS")
    aa("--epochs", type=int, default=1, help="number of epochs to train")
    aa("--context_length", type=int, help="past context length")
    aa("--prediction_length", type=int, help="future prediction length")
    aa("--skip_size", type=int, help="LSTNet skip size")
    aa("--ar_window", type=int, help="LSTNet AR window linear part")
    aa("--rnn_num_layers", type=int, help="LSTNet RNN (GRU) number of layers")
    aa("--skip_rnn_num_layers", type=int, help="LSTNet skip RNN (GRU) number of layers")
    aa("--channels", type=int, help="number of channels for first conv1d layer")
    aa("--seed", type=int, default=12, help="RNG seed")
    args = parser.parse_args()
    logger.info(f"Passed arguments: {args}")

    dataset = load_dataset(args.dataset_name, Path(args.dataset_path))
    logger.info(f"Train data shape: {next(iter(dataset.train))['target'].shape}")
    logger.info(f"Test data shape: {next(iter(dataset.test))['target'].shape}")

    predictor = train(
        dataset,
        args.output_dir,
        args.model_dir,
        args.epochs,
        args.context_length,
        args.prediction_length,
        args.skip_size,
        args.ar_window,
        args.rnn_num_layers,
        args.skip_rnn_num_layers,
        args.channels,
        args.seed,
    )
    # store serialized model artifacts
    save(predictor, args.model_dir)
    logger.info(f"Model serialized in {args.model_dir}")

    forecasts, tss, agg_metrics, item_metrics = evaluate(predictor, dataset.test)

    logger.info(f"Root Relative Squared Error (RSE): {rse(agg_metrics, dataset.test)}")

    with open(
        osp.join(args.output_dir, "train_agg_metrics.json"), "w", encoding="utf-8"
    ) as fout:
        json.dump(agg_metrics, fout)

    item_metrics.to_csv(
        osp.join(args.output_dir, "item_metrics.csv.gz"),
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
