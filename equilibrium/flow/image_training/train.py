import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import datasets
from tqdm import tqdm
from equilibrium.flow.path.path import ProbPath
from equilibrium.flow.path.affine import CondOTProbPath
from equilibrium.models.unet import UNetModel
from equilibrium.utils import save_model, load_model


MODEL_PATH = "output/ckpt"


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

    ds = datasets.load_dataset('Maysee/tiny-imagenet', split='train[:10%]')
    batch = next(ds.iter(batch_size=bsz))
    samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
    labels = jnp.array(batch["label"])

    t_rngs = nnx.Rngs(108)
    # loss, grads = grad_fn(model, path, samples, labels, t_rngs)
    # loss.block_until_ready()

    # optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-2))
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=1e-3))
    for epoch in range(50):
        epoch_losses = []
        pbar = tqdm(ds.iter(batch_size=bsz), total=ds.shape[0] // bsz)
        for batch in pbar:
            try:
                samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
                labels = jnp.array(batch["label"])
                loss = train_step(model, path, samples, labels, optimizer, t_rngs)
                epoch_losses.append(loss.item())
                pbar.set_description(f"epoch {epoch}: loss = {loss}")
            except:
                print(f"Failed to process batch #{pbar.n}")
        pbar.write(
            f"==> epoch {epoch}: avg loss = {jnp.array(epoch_losses).mean()}"
        )
        save_model(model, MODEL_PATH + f"/{epoch}")




class Flow(nnx.Module):
    def __init__(self, model):
        self.model = model

    def step(self, x_t: jax.Array, t_start: jax.Array, t_end: jax.Array) -> jax.Array:
        # t_start = t_start.reshape(1, 1)
        t_start = jnp.repeat(t_start, x_t.shape[0], axis=0)
        # new_t = t_start + (t_end - t_start) / 2
        # new_x_t = x_t + self.model(x_t, t_start, {}) * (t_end - t_start) / 2
        # return x_t + (t_end - t_start) * self.model(new_x_t, new_t, {})


        return x_t + (t_end - t_start) * self.model(
            extra={},
            timesteps=t_start + (t_end - t_start) / 2,
            x=x_t + self.model(
                x=x_t,
                timesteps=t_start,
                extra={}
            ) * (t_end - t_start) / 2)

        # return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)




def sampling():
    model = load_model(lambda: UNetModel(3, rngs=nnx.Rngs(28)), MODEL_PATH, to_cpu=True)
    flow = Flow(model)
    rngs = nnx.Rngs(113)
    x = jax.random.normal(rngs(), (1, 64, 64, 3))
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = jnp.linspace(0, 1.0, n_steps + 1)

    # axes[0].scatter(x[:, 0], x[:, 1], s=10)
    # axes[0].set_title(f't = {time_steps[0]:.2f}')
    # axes[0].set_xlim(-3.0, 3.0)
    # axes[0].set_ylim(-3.0, 3.0)

    samples = []
    for i in tqdm(range(n_steps)):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        samples.append(x)
    samples = jnp.vstack(samples)
    plot_samples(samples, "output/generated.jpg")


    # plt.tight_layout()
    # plt.savefig("output/generated.jpg")



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
        # print(f"r = {r}, c = {c}")
        axs[r, c].imshow(sample)
    plt.tight_layout()
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



# TODO (assuming implementation is correct):
# 1. Clean code
# 2. Implement char-level one-hot-encoded transformer
# 3. Train char diffusion