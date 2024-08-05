import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm import create_model, list_models
from mup import set_base_shapes
from sklearn.metrics import top_k_accuracy_score
from pytorch_fob.tasks import TaskModel
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_warn, log_info
from pytorch_fob.optimizers import Optimizer
from .scaling_vit import WidthScalingVisionTransformer

class ImagenetModel(TaskModel):
    def __init__(self, optimizer: Optimizer, config: TaskConfig):
        model = self._create_model(config)
        super().__init__(model, optimizer, config)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, labels = batch["image"], batch["label"]
        return self.model(imgs), labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        preds, labels = self.forward(batch)
        loss = self.compute_and_log_loss(preds, labels, "train")
        self.compute_and_log_acc(preds, labels, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels = self.forward(batch)
        self.compute_and_log_loss(preds, labels, "val")
        self.compute_and_log_acc(preds, labels, "val")

    def test_step(self, batch, batch_idx):
        preds, labels = self.forward(batch)
        self.compute_and_log_acc(preds, labels, "test")

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, stage: str) -> dict[str, float]:
        pred_probs = preds.softmax(-1).detach().cpu().numpy()
        gts = labels.detach().cpu().numpy()
        all_labels = np.arange(1000)
        top_1_acc = top_k_accuracy_score(y_true=gts, y_score=pred_probs, k=1, labels=all_labels)
        top_5_acc = top_k_accuracy_score(y_true=gts, y_score=pred_probs, k=5, labels=all_labels)
        self.log(f"{stage}_top1_acc", top_1_acc, sync_dist=True)
        self.log(f"{stage}_top1_err", 1 - top_1_acc, sync_dist=True)
        self.log(f"{stage}_top5_acc", top_5_acc, sync_dist=True)
        self.log(f"{stage}_top5_err", 1 - top_5_acc, sync_dist=True)
        return {"top1": top_1_acc, "top5": top_5_acc}

    def compute_and_log_loss(self, preds: torch.Tensor, labels: torch.Tensor, stage: str) -> torch.Tensor:
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        return loss

    def _create_model(self, config: TaskConfig):
        model_name: str = config.model.name

        # create model
        if model_name == "width_scaling_vit":
            mup_scaling = config.model.get("mup_scaling", True)
            model = WidthScalingVisionTransformer(
                width=config.model.width,
                replace_head=mup_scaling,
                **config.model.kwargs_timm,
            )
            # μP-scaling is active by default in width_scaling_vit, but we can disable it via config
            if mup_scaling:
                base_model = WidthScalingVisionTransformer(width=1, **config.model.kwargs_timm)
                delta_model = WidthScalingVisionTransformer(width=2, **config.model.kwargs_timm)
                set_base_shapes(model, base_model, delta=delta_model)
                model.reset_weights()
        else:
            try:
                model = create_model(model_name, **config.model.kwargs_timm)
            except RuntimeError as e:
                available_models = list_models()
                log_info(f"Available Models are {available_models}")
                raise ValueError("Unsupported model given.") from e

        # taking care of model specific changes
        if model_name == "wide_resnet50_2":
            # 7x7 conv might be pretty large for 64x64 images
            model.conv1 = nn.Conv2d(3,  # rgb color
                                    64,
                                    kernel_size=config.model.stem.kernel_size,
                                    stride=config.model.stem.stride,
                                    padding=config.model.stem.padding,
                                    bias=False
                                    )
            # pooling small images might be bad
            if not config.model.maxpool:
                model.maxpool = nn.Identity()  # type:ignore

        elif model_name == "davit_tiny.msft_in1k":
            # msft_in1k: trained on imagenet 1k by authors
            # https://huggingface.co/timm/davit_tiny.msft_in1k
            if config.model.stem == "default":
                # off the shelf DaVit
                pass
            elif config.model.stem == "wrn_conv":
                model.stem = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
                    LayerNorm2d((96,))
                )
            elif config.model.stem == "custom_conv":
                model.stem = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=96, kernel_size=15, stride=1, padding=3),
                    LayerNorm2d((96,))
                )
            else:
                log_warn(f"stem argument '{config.model.stem}' unknown to classification task.")

        return model


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors,
    thanks to https://github.com/dingmyu/davit/blob/main/timm/models/layers/norm.py """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
