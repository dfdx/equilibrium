import argparse
import gc
import logging
import math
from typing import Iterable

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from equilibrium.flow.path.affine import CondOTProbPath # , MixtureDiscreteProbPath
# from equilibrium.flow.path.scheduler import PolynomialConvexScheduler
# from models.ema import EMA
# from torch.nn.parallel import DistributedDataParallel
# from torchmetrics.aggregation import MeanMetric
# from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: jax.Device) -> jax.Array:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = jax.random.normal(jax.random.PRNGKey(0), (num_samples,))
    sigma = jnp.exp(rnd_normal * P_std + P_mean)
    time = 1 / (1 + sigma)
    time = jnp.clip(time, a_min=0.0001, a_max=1.0)
    return time


def train_one_epoch(
    model: nnx.Module,
    data_loader: Iterable,
    optimizer: nnx.Optimizer,
    # lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    # loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    *,
    rngs: nnx.Rngs = nnx.Rngs(0),
):
    # gc.collect()
    # model.train(True)
    # batch_loss = MeanMetric().to(device, non_blocking=True)
    # epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.discrete_flow_matching:
        raise ValueError("not supported")
        # scheduler = PolynomialConvexScheduler(n=3.0)
        # path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        if args.discrete_flow_matching:
            raise ValueError("not supported")
            # samples = (samples * 255.0).to(torch.long)
            # t = torch.torch.rand(samples.shape[0]).to(device)

            # # sample probability path
            # x_0 = (
            #     torch.zeros(samples.shape, dtype=torch.long, device=device) + MASK_TOKEN
            # )
            # path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

            # # discrete flow matching loss
            # logits = model(path_sample.x_t, t=t, extra=conditioning)
            # loss = torch.nn.functional.cross_entropy(
            #     logits.reshape([-1, 257]), samples.reshape([-1])
            # ).mean()
        else:
            # Scaling to [-1, 1] from [0, 1]
            samples = samples * 2.0 - 1.0
            noise = jax.random.normal(rngs(), shape=samples.shape, dtype=samples.dtype)
            if args.skewed_timesteps:
                raise ValueError("not supported")
                # t = skewed_timestep_sample(samples.shape[0], device=device)
            else:
                # t = torch.torch.rand(samples.shape[0]).to(device)
                t = jax.random.uniform(rngs(), samples.shape[0], dtype=samples.dtype)
            path_sample = path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}