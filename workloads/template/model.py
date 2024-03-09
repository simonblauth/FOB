import torch
from workloads import WorkloadModel
from engine.configs import WorkloadConfig
from submissions import Submission


class TemplateModel(WorkloadModel):
    def __init__(self, submission: Submission, workload_config: WorkloadConfig):
        # Here you can see some examples on how to include the config
        # 1) parameters that should change depending on the experiment are placed in the default.yaml
        hidden_channels_from_yaml = workload_config.model.hidden_channels

        # 2) you can also add other type, e.g. activation function, but this usually needs some code
        if workload_config.model.activation.lower() == "ReLU".lower():
            self.activation = torch.nn.ReLU
        # 3) the workload_config is a dict, you could access the values just like a dict; we prefer the dots
        elif workload_config["model"]["activation"].lower() == "GELU".lower():
            self.activation = torch.nn.GELU
        else:
            raise NotImplementedError(f"{workload_config.model.activation} is not yet supported for {type(self)}")

        model = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels_from_yaml),
            self.activation(),
            torch.nn.Linear(hidden_channels_from_yaml, 1),
            self.activation(),
        )
        self.loss = torch.nn.functional.mse_loss
        super().__init__(model, submission, workload_config)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.compute_and_log_loss(batch, "val_loss")

    def test_step(self, batch, batch_idx):
        self.compute_and_log_loss(batch, "test_loss")

    def compute_and_log_loss(self, batch, log_name: str):
        x = y = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log(log_name, loss)
