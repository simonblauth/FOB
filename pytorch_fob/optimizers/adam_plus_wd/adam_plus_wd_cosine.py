from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    # _capturable_doc,
    _default_to_fused_or_foreach,
    # _differentiable_doc,
    _dispatch_sqrt,
    # _foreach_doc,
    # _fused_doc,
    _get_value,
    # _maximize_doc,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    params_t,
)
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

__all__ = ["AdamPlus", "adamplus"]

# TODO: group_params function (set wd to 0 if not regularized instead of extra group params)


class AdamPlus(Optimizer):
    """
    Initializes the AdamPlusWD optimizer.

    Args:
        params (params_t): The parameters to optimize.
        train_steps (int): The number of training steps.
        lr_grad (Union[float, Tensor], optional): The learning rate gradient during the warmup phase. Defaults to 1e-6.
        lr_decay (float, optional): How much to decay the learning rate as a factor of the peak value. Defaults to 0.01.
        betas (Tuple[float, float], optional): The beta parameters. Defaults to (0.9, 0.999).
        eps (float, optional): The epsilon value. Defaults to 1e-8.
        reg_step_size (int, optional): The interval at which the detection of the l2 inflection point takes place. Defaults to 100.
        weight_decay (float, optional): The weight decay. Defaults to 1e-2.
        amsgrad (bool, optional): Whether to use AMSGrad. Defaults to False.

    References:
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params: params_t,
        train_steps: int,
        lr_grad: Union[float, Tensor] = 1e-6,
        lr_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        reg_step_size: int = 100,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr_grad:
            raise ValueError(f"Invalid learning rate gradient: {lr_grad}")
        if isinstance(lr_grad, Tensor) and foreach and not capturable:
            raise ValueError("lr_grad as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.reg_step_size = reg_step_size
        self.train_steps = train_steps

        defaults = dict(
            lr_grad=lr_grad,
            lr_decay=lr_decay,
            train_steps=train_steps,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
            group.setdefault("lr_update", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        lrs,
        lr_indices,
        prev_regs,
        prev_reg_gradients,
        prev_reg_second_derivatives,
        decay_steps,
        min_lrs,
        state_steps,
    ):
        for p, lr_idx in zip(group["params"], group["lr_index"]):
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            lr_indices.append(lr_idx)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros((), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # AdamPlus stuff
                state["lr"] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state["prev_reg"] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state["prev_gradient"] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                state["prev_second_derivative"] = torch.tensor([0.0], dtype=torch.float, device=p.device)
                # is `-1` during warmup phase, and then contains the starting step of the decay phase after it starts
                state["decay_step"] = torch.tensor([-1], dtype=torch.float, device=p.device)
                # the minimum lr to decay towards
                state["min_lr"] = torch.tensor([0.0], dtype=torch.float, device=p.device)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            lrs.append(state["lr"])
            prev_regs.append(state["prev_reg"])
            prev_reg_gradients.append(state["prev_gradient"])
            prev_reg_second_derivatives.append(state["prev_second_derivative"])
            decay_steps.append(state["decay_step"])
            min_lrs.append(state["min_lr"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError("`requires_grad` is not supported for `step` in differentiable mode")

            # Foreach without capturable does not support a tensor lr_grad
            # TODO: is this even the case for adamplus?
            if group["foreach"] and isinstance(group["lr_grad"], Tensor) and not group["capturable"]:
                raise RuntimeError("lr_grad as a Tensor is not supported for capturable=False and foreach=True")

            state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # TODO: is mean lr appropriate?
        is_initialized = False
        lr_index_dict = {}
        for group in self.param_groups:
            if group["lr_update"]:
                for p, lr_index in zip(group["params"], group["lr_index"]):
                    state = self.state[p]
                    if "lr" in state:
                        lr = self.state[p]["lr"]
                        lr_index_dict[lr_index] = lr
                        is_initialized = True
        if is_initialized:
            mean_lr = sum(lr_index_dict.values()) / len(lr_index_dict)
            for group in self.param_groups:
                if not group["lr_update"]:
                    for p, lr_index in zip(group["params"], group["lr_index"]):
                        if lr_index == -1:
                            self.state[p]["lr"] = mean_lr
                        else:
                            self.state[p]["lr"].copy_(lr_index_dict[lr_index])

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            lrs = []
            lr_indices = []
            prev_regs = []
            prev_reg_gradients = []
            prev_reg_second_derivatives = []
            decay_steps = []
            min_lrs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lrs,
                lr_indices,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                decay_steps,
                min_lrs,
                state_steps,
            )

            adamplus(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                lrs,
                prev_regs,
                prev_reg_gradients,
                prev_reg_second_derivatives,
                decay_steps,
                min_lrs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                lr_update=group["lr_update"],
                lr_grad=group["lr_grad"],
                lr_decay=group["lr_decay"],
                train_steps=self.train_steps,
                reg_step_size=self.reg_step_size,
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def adamplus(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    lrs: List[Tensor],
    prev_regs: List[Tensor],
    prev_reg_gradients: List[Tensor],
    prev_reg_second_derivatives: List[Tensor],
    decay_steps: List[Tensor],
    min_lrs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    lr_update: bool,
    lr_grad: Union[float, Tensor],
    lr_decay: float,
    train_steps: int,
    reg_step_size: int,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adamw
        # TODO: fused support possible?
        raise RuntimeError("Fused Adam is not supported for AdamPlus")
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        lrs,
        prev_regs,
        prev_reg_gradients,
        prev_reg_second_derivatives,
        decay_steps,
        min_lrs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        lr_update=lr_update,
        lr_grad=lr_grad,
        lr_decay=lr_decay,
        train_steps=train_steps,
        reg_step_size=reg_step_size,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    lrs: List[Tensor],
    prev_regs: List[Tensor],
    prev_reg_gradients: List[Tensor],
    prev_reg_second_derivatives: List[Tensor],
    decay_steps: List[Tensor],
    min_lrs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    lr_update: bool,
    lr_grad: Union[float, Tensor],
    lr_decay: float,
    train_steps: int,
    reg_step_size: int,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        lr = lrs[i]
        prev_reg = prev_regs[i]
        prev_reg_gradient = prev_reg_gradients[i]
        prev_reg_second_derivative = prev_reg_second_derivatives[i]
        decay_step = decay_steps[i]
        min_lr = min_lrs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (param.is_cuda and step_t.is_cuda) or (
                param.is_xla and step_t.is_xla
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size.item())

        if lr_update:
            if decay_step > 0:
                # cosine update
                cur_step = step - decay_step
                T_max = train_steps - decay_step
                factor = (1 + torch.cos(torch.pi * cur_step / T_max)) / (
                    1 + torch.cos(torch.pi * (cur_step - 1) / T_max)
                )
                lr.sub_(min_lr).mul_(factor).add_(min_lr)
            elif step % reg_step_size == 0 and decay_step == -1:
                current_l2m = param.square().mean()
                if step > reg_step_size:
                    current_reg_gradient = current_l2m - prev_reg
                if step > reg_step_size * 2:
                    current_reg_second_derivative = current_reg_gradient - prev_reg_gradient

                # Peak detection for gradient
                if (
                    step > reg_step_size * 3
                    and prev_reg_gradient > current_reg_gradient
                    and prev_reg_second_derivative > 0
                    and current_reg_second_derivative <= 0
                ):
                    decay_step.copy_(step)
                    min_lr.copy_(lr * lr_decay)

                # Update previous values for next iteration
                prev_reg.copy_(current_l2m)
                if step > reg_step_size:
                    prev_reg_gradient.copy_(current_reg_gradient)
                if step > reg_step_size * 2:
                    prev_reg_second_derivative.copy_(current_reg_second_derivative)

                lr.add_(lr_grad)
            elif decay_step == -1:
                lr.add_(lr_grad)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    lrs: List[Tensor],
    prev_regs: List[Tensor],
    prev_reg_gradients: List[Tensor],
    prev_reg_second_derivatives: List[Tensor],
    decay_steps: List[Tensor],
    min_lrs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    lr_update: bool,
    lr_grad: Union[float, Tensor],
    lr_decay: float,
    train_steps: int,
    reg_step_size: int,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    if len(params) == 0:
        return

    # TODO: figure out how relevant this is
    # if isinstance(lr, Tensor) and not capturable:
    #     raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            lrs,
            prev_regs,
            prev_reg_gradients,
            prev_reg_second_derivatives,
            decay_steps,
            min_lrs,
            state_steps,
        ]
    )
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_lrs,
        device_prev_regs,
        device_prev_reg_gradients,
        device_prev_reg_second_derivatives,
        device_decay_steps,
        device_min_lrs,
        device_state_steps,
    ), _ in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
        device_max_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_max_exp_avg_sqs]
        device_lrs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lrs]
        device_prev_regs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_prev_regs]
        device_prev_reg_gradients = [
            torch.view_as_real(x) if torch.is_complex(x) else x for x in device_prev_reg_gradients
        ]
        device_prev_reg_second_derivatives = [
            torch.view_as_real(x) if torch.is_complex(x) else x for x in device_prev_reg_second_derivatives
        ]
        device_decay_steps = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_decay_steps]
        device_min_lrs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_min_lrs]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            tmp_lrs = [1 - lr * weight_decay for lr in device_lrs]
            torch._foreach_mul_(device_params, tmp_lrs)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr.item() / bc) * -1 for lr, bc in zip(device_lrs, bias_correction1)])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)

        if lr_update:
            # warmup phase
            if any([decay_step == -1 for decay_step in device_decay_steps]):
                if device_state_steps[0] % reg_step_size == 0:
                    square_params = torch._foreach_pow(device_params, 2)

                    current_l2ss = [square_param.sum() for square_param in square_params]
                    if device_state_steps[0] > reg_step_size:
                        current_gradients = torch._foreach_sub(current_l2ss, device_prev_regs)
                    if device_state_steps[0] > reg_step_size * 2:
                        current_reg_second_derivatives = torch._foreach_sub(
                            current_gradients, device_prev_reg_gradients
                        )

                    # Peak detection for gradient
                    if device_state_steps[0] > reg_step_size * 3:
                        for i in range(len(device_params)):
                            if device_prev_reg_gradients[i] > current_gradients[i] and device_decay_steps[i] == -1:
                                device_decay_steps[i].copy_(device_state_steps[0])
                                device_min_lrs[i].copy_(device_lrs[i] * lr_decay)

                    # Update previous values for next iteration
                    torch._foreach_copy_(device_prev_regs, current_l2ss)
                    if device_state_steps[0] > reg_step_size:
                        torch._foreach_copy_(device_prev_reg_gradients, current_gradients)
                    if device_state_steps[0] > reg_step_size * 2:
                        torch._foreach_copy_(device_prev_reg_second_derivatives, current_reg_second_derivatives)

                for i in range(len(device_decay_steps)):
                    if device_decay_steps[i] == -1:
                        device_lrs[i].add_(lr_grad)

            # cosine decay phase
            if any([decay_step > 0 for decay_step in device_decay_steps]):
                for i in range(len(device_decay_steps)):
                    if device_decay_steps[i] > 0:
                        cur_step = device_state_steps[i] - device_decay_steps[i]
                        T_max = train_steps - device_decay_steps[i]
                        factor = (1 + torch.cos(torch.pi * cur_step / T_max)) / (
                            1 + torch.cos(torch.pi * (cur_step - 1) / T_max)
                        )
                        device_lrs[i].sub_(device_min_lrs[i]).mul_(factor).add_(device_min_lrs[i])


def _fused_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    grad_scale_dict = {grad_scale.device: grad_scale} if grad_scale is not None else None
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict = {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )
    for (device, _), (
        (
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
        ),
        _,
    ) in grouped_tensors.items():
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adamw_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))
