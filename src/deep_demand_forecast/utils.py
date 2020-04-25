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
    model: Predictor, test_data: ListDataset, num_samples: int = 1
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
