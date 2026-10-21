"""Experimental univariate adapter for the timesfm 3.0.2 PyTorch API.

Downloaded 3.0 weights are restricted to non-commercial, non-production use.
Keep imports lazy so the core application does not require model dependencies.
"""

import numpy as np


class TimesFM3Adapter:
    def __init__(self, config, device):
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.forecaster = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=config.model_id,
            per_core_batch_size=config.batch_size,
            device="cuda:0" if device == "cuda" else "cpu",
        ))
        if tuple(self.forecaster.config.quantiles) != tuple(i / 10 for i in range(1, 10)):
            raise RuntimeError("TimesFM 3.0 checkpoint must provide quantiles q10 through q90")
        if config.effective_context > self.forecaster.global_context:
            raise RuntimeError("TimesFM 3.0 checkpoint cannot cover the requested context")

    @property
    def model(self):
        return self.forecaster.model

    @property
    def device(self):
        return self.forecaster.device

    def forecast(self, horizon, inputs):
        # Each origin is an independent 1D series, never a multivariate channel.
        outputs = list(self.forecaster.predict_batch(
            contexts=inputs, horizon=horizon, return_quantiles=True,
            use_symmetric_averaging=True, make_positive=True, sort_quantiles=True,
            use_znorm=False, padding_mode="none",
        ))
        if len(outputs) != len(inputs):
            raise RuntimeError("TimesFM 3.0 returned a different number of forecasts than inputs")
        for item in outputs:
            if np.shape(item.forecast) != (horizon,) or np.shape(item.quantiles) != (horizon, 9):
                raise RuntimeError("Unexpected TimesFM 3.0 point or quantile forecast shape")
        point = np.stack([item.forecast for item in outputs])
        quantiles = np.stack([item.quantiles for item in outputs])
        # Normalize to the existing runtime contract: point channel then deciles.
        return point, np.concatenate([point[:, :, None], quantiles], axis=2)
