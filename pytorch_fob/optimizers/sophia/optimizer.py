"""
Sophia optimizer proposed in: https://arxiv.org/abs/2305.14342
"""

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler
from pytorch_fob.optimizers.sophia.sophia import SophiaG


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
    optimizer = SophiaG(
        params=parameter_groups,
        lr=lr,
        rho=config.rho,
        betas=(config.beta1, config.beta2),
        weight_decay=weight_decay,
    )
    scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_scheduler.interval
        }
    }
