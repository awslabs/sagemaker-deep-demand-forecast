import os
import sys

# resolves pytest not finding parent module issue
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np

from gluonts.dataset.common import TrainDatasets, save_datasets
from gluonts.dataset.artificial import default_synthetic

from data import load_multivariate_datasets
from train import train, save
from inference import model_fn, transform_fn
from utils import evaluate


def create_multivariate_datasets(data_dir: str) -> None:
    info, train_ds, test_ds = default_synthetic()
    save_datasets(TrainDatasets(metadata=info.metadata, train=train_ds, test=test_ds), data_dir)
    return


def test(tmpdir) -> None:
    data_dir = tmpdir.mkdir("data_dir")
    output_dir = tmpdir.mkdir("output_dir")
    model_dir = tmpdir.mkdir("model_dir")

    create_multivariate_datasets(data_dir)
    datasets = load_multivariate_datasets(data_dir)
    prediction_length = 2

    predictor = train(
        datasets,
        output_dir,
        model_dir,
        context_length=12,
        prediction_length=prediction_length,
        skip_size=2,
        ar_window=3,
        channels=6,
        scaling=False,
        output_activation="sigmoid",
        epochs=1,
        batch_size=5,
        learning_rate=1e-2,
        seed=42,
    )

    forecasts, tss, agg_metrics, item_metrics = evaluate(predictor, datasets.test, num_samples=1)

    save(predictor, model_dir)

    predictor = model_fn(model_dir)

    request_body = {}
    request_body["target"] = np.random.randn(10, prediction_length).tolist()
    request_body["start"] = "2001-01-01"
    request_body["source"] = []
    ret, _ = transform_fn(predictor, json.dumps(request_body), None, None)
    forecast_samples = np.array(ret["forecasts"]["samples"])
    assert forecast_samples.shape == (1, prediction_length, 10)
    agg_metrics = json.loads(ret["agg_metrics"])
    for metric in ["RMSE", "ND", "MSE"]:
        assert agg_metrics[metric] < 1.5, f"assertion failed for metric: {metric}"
    return
