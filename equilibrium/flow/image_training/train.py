import os
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import datasets
import orbax.checkpoint as ocp
from tqdm import tqdm
from equilibrium.flow.path.path import ProbPath
from equilibrium.flow.path.affine import CondOTProbPath
from equilibrium.models.unet import UNetModel


def loss_fn(model: nnx.Module, path: ProbPath, samples: jax.Array, labels: jax.Array, rngs: nnx.Rngs):
    samples = samples * 2.0 - 1.0
    noise = jax.random.normal(rngs(), shape=samples.shape, dtype=samples.dtype)
    t = jax.random.uniform(rngs(), samples.shape[0], dtype=samples.dtype)
    path_sample = path.sample(t=t, x_0=noise, x_1=samples)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t
    loss = jnp.pow(model(x_t, t, extra={"labels": labels}) - u_t, 2).mean()
    return loss


# # autoencoder loss (for test)
# def loss_fn(model: nnx.Module, path: ProbPath, samples: jax.Array, labels: jax.Array, rngs: nnx.Rngs):
#     samples = samples * 2.0 - 1.0
#     # noise = jax.random.normal(rngs(), shape=samples.shape, dtype=samples.dtype)
#     t = jax.random.uniform(rngs(), samples.shape[0], dtype=samples.dtype)
#     # path_sample = path.sample(t=t, x_0=noise, x_1=samples)
#     # x_t = path_sample.x_t
#     # u_t = path_sample.dx_t
#     # loss = jnp.pow(model(x_t, t, extra={"labels": labels}) - u_t, 2).mean()
#     loss = jnp.pow(model(samples, t, {}) - samples, 2).mean()
#     return loss



@nnx.jit(static_argnums=(1,))
def train_step(model: nnx.Module, path: ProbPath, samples: jax.Array, labels: jax.Array, optimizer: nnx.Optimizer, rngs: nnx.Rngs):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, path, samples, labels, rngs)
    optimizer.update(grads)
    return loss


def main():
    rngs = nnx.Rngs(0)
    path = CondOTProbPath()
    bsz, in_channels, hw = 2, 3, 64
    model = UNetModel(in_channels, rngs=rngs)
    # samples = jax.random.normal(rngs(), (bsz, hw, hw, in_channels))
    # labels = jax.random.randint(rngs(), bsz, 0, 3)
    # timesteps = jax.random.uniform(rngs(), bsz)
    # out = model(samples, timesteps, {"labels": labels})

    # loss = loss_fn(model, path, samples, labels, rngs)

    # grad_fn = nnx.value_and_grad(loss_fn)
    # grad_fn = nnx.jit(grad_fn, static_argnums=(1,))
    # loss, grads = grad_fn(model, path, samples, labels, nnx.Rngs(5))

    ds = datasets.load_dataset('Maysee/tiny-imagenet', split='train[:100]')
    batch = next(ds.iter(batch_size=bsz))
    samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
    labels = jnp.array(batch["label"])

    t_rngs = nnx.Rngs(108)
    # loss, grads = grad_fn(model, path, samples, labels, t_rngs)
    # loss.block_until_ready()

    # optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3))
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=1e-3))
    for epoch in range(100):
        epoch_losses = []
        pbar = tqdm(ds.iter(batch_size=bsz), total=ds.shape[0] // bsz)
        for batch in pbar:
            samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
            labels = jnp.array(batch["label"])
            loss = train_step(model, path, samples, labels, optimizer, t_rngs)
            epoch_losses.append(loss.item())
            pbar.set_description(f"epoch {epoch}: loss = {loss}")
        pbar.write(
            f"==> epoch {epoch}: avg loss = {jnp.array(epoch_losses).mean()}"
        )




def save_model(model, ckpt_dir):
    _, state = nnx.split(model)
    nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.join(ckpt_dir, 'state'), state)


###############################################################################

from jax.experimental.ode import odeint
from matplotlib import pyplot as plt


def dummy_model(y, t, extras):
    print(f"y.shape = {y.shape}, t.shape = {t.shape}")
    return y

def adapted_time(model):
    def wrapper(y, t):
        ta = jnp.repeat(t, y.shape[0], axis=0)
        return model(y, ta, {})
    return wrapper


def generate(model: UNetModel, shape: tuple, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
    noise = jax.random.normal(rngs(), shape=shape, dtype=dtype)
    t = jnp.linspace(0, 1)
    wrapper = adapted_time(model)
    out = odeint(wrapper, noise, t)
    return out


def plot_samples(samples: jax.Array, path: str | None = None):
    """
    Arguments:
    ----------
    samples : jax.Array
        Input image data of shape (B, H, W, C)
    """
    length = samples.shape[0]
    n_rows = int(jnp.sqrt(length))
    n_cols = int(jnp.ceil(length / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(length):
        sample = samples[i, :, :, :]
        sample = sample - sample.min()
        sample = sample / sample.max()
        r, c = i // n_cols, i % n_cols
        print(f"r = {r}, c = {c}")
        axs[r, c].imshow(sample)
    if path:
        fig.savefig(path)



def main_inf():
    rngs = nnx.Rngs(0)
    out = generate(model, (8, 64, 64, 3))
    for i in range(8):
        plot_samples(out[:, i, :, :, :], f"output/samples_{i}.jpg")
        plt.clf()


if __name__ == "__main__" and "__file__" in globals():
    main()