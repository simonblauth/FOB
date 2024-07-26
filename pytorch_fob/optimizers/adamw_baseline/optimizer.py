from torch.optim import AdamW
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from mup import MuAdamW
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
    optim_cls = MuAdamW if config.mup_scaling else AdamW
    optimizer = optim_cls(
        parameter_groups,
        lr=lr,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2),
        weight_decay=weight_decay,
        foreach=config.foreach,
    )
    lr_scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": config.lr_scheduler.interval
        }
    }
