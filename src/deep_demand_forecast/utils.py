from typing import List, Dict, Tuple, Any
import os
import logging
from logging.config import fileConfig

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Forecast


def get_logger(config_path) -> logging.Logger:
    fileConfig(config_path)
    logger = logging.getLogger()
    return logger


def evaluate(
    model: Predictor, test_data: ListDataset, num_samples: int = 10
) -> Tuple[List[Forecast], List[pd.Series], Dict[str, float], pd.DataFrame]:
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=model, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = MultivariateEvaluator()
    agg_metrics, item_metrics = evaluator(
        iter(tss), iter(forecasts), num_series=len(test_data)
    )
    return forecasts, tss, agg_metrics, item_metrics


def compare_two_item_metrics(
    item_metrics: pd.DataFrame, col_a: str, col_b: str, path: str
) -> None:
    matplotlib.use("Agg")
    figs, axes = plt.subplots(3)
    item_metrics[col_a].plot(kind="hist", ax=axes[0], figsize=(10, 15))
    axes[0].set_xlabel(col_a)
    item_metrics[col_b].plot(kind="hist", ax=axes[1])
    axes[1].set_xlabel(col_b)
    item_metrics.plot(x=col_a, y=col_b, kind="scatter", ax=axes[2])
    plt.grid(which="both")
    figs.savefig(os.path.join(path, f"{col_a}_vs_{col_b}.png"))
    return
