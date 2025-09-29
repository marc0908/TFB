import torch
from ts_benchmark.baselines.time_series_library.models.TimeXer import Model as TimeXerModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_model": 512,
    "d_ff": 2048,
    "embed": "timeF",
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "dropout": 0.1,
    "batch_size": 64,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
    "use_norm": True,
    "features": "M",
    "task_name": "short_term_forecast",
}


class TimeXer(DeepForecastingModelBase):
    """
    TimeXer adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the TimeXerModel.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(TimeXer, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeXer"

    def _init_model(self):
        return TimeXerModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        # decoder input
        dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float()
        dec_input = (
            torch.cat([target[:, : self.config.label_len, :], dec_input], dim=1)
            .float()
            .to(input.device)
        )
        output = self.model(input, input_mark, dec_input, target_mark)

        return {"output": output}