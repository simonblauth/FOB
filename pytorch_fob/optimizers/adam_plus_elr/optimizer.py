from inspect import Parameter
from typing import Any

from lightning.pytorch.utilities.types import OptimizerLRScheduler

from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel, ParameterGroup
from pytorch_fob.optimizers.adam_plus_elr.adam_plus_elr import AdamPlus


def to_optimizer_dict(
        pg: ParameterGroup,
        lr_index_master: dict[str, int],
        weight_decay: float,
        lr_grad: float,
    ) -> dict[str, list[Parameter] | Any]:
    names = sorted(pg.named_parameters)
    # is_master = pg.weight_decay_multiplier is None or pg.weight_decay_multiplier > 0
    d = {
        "params": [pg.named_parameters[n] for n in names],
        "names": names,
        "lr_update": True,
        "lr": 0,  # not used, just so LRMonitor doesn't crash
        "weight_decay": pg.weight_decay_multiplier * weight_decay \
                if pg.weight_decay_multiplier is not None else weight_decay,
        "lr_grad": pg.lr_multiplier * lr_grad if pg.lr_multiplier is not None else lr_grad,
        **pg.optimizer_kwargs,
    }
    lr_index = []
    for n in names:
        if n in lr_index_master:
            lr_index.append(lr_index_master[n])
        elif n.replace("bias", "weight") in lr_index_master:
            lr_index.append(lr_index_master[n.replace("bias", "weight")])
        else:
            lr_index.append(-1)
    d["lr_index"] = lr_index
    return d


def fill_master_dict(pgs: list[ParameterGroup]) -> dict[str, int]:
    lr_index_master = {}
    for pg in pgs:
        is_master = pg.weight_decay_multiplier is None or pg.weight_decay_multiplier > 0
        if is_master:
            for n in pg.named_parameters:
                lr_index_master[n] = len(lr_index_master)
    return lr_index_master


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr_grad=config.lr_grad
    weight_decay = config.weight_decay
    parameter_groups = model.parameter_groups()
    lr_index_master = fill_master_dict(parameter_groups)
    params_fob = [to_optimizer_dict(pg, lr_index_master, weight_decay, lr_grad) for pg in parameter_groups]
    optimizer = AdamPlus(
        params=params_fob,
        train_steps=config.max_steps,
        lr_grad=lr_grad,
        lr_decay=config.lr_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        alpha=config.alpha,
        weight_decay=weight_decay,
        foreach=config.foreach,
    )
    return optimizer
