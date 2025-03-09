# based on: https://github.com/facebookresearch/flow_matching/blob/main/examples/standalone_flow_matching.ipynb

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.datasets import make_moons


class Flow(nnx.Module):
    def __init__(self, dim: int = 2, h: int = 64, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.net = nnx.Sequential(
            nnx.Linear(dim + 1, h, rngs=rngs), nnx.elu,
            nnx.Linear(h, h, rngs=rngs), nnx.elu,
            nnx.Linear(h, h, rngs=rngs), nnx.elu,
            nnx.Linear(h, dim, rngs=rngs))

    def __call__(self, x_t: jax.Array, t: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate((t, x_t), axis=-1))

    def step(self, x_t: jax.Array, t_start: jax.Array, t_end: jax.Array) -> jax.Array:
        t_start = t_start.reshape(1, 1)
        t_start = jnp.repeat(t_start, x_t.shape[0], axis=0)
        return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)



@nnx.jit
def train_step(model, optimizer, prng):
    def loss_fn(model, x_t, t, dx_t):
        return optax.l2_loss(model(x_t=x_t, t=t), dx_t).mean()
    prng_x_0, prng_t = jax.random.split(prng)
    # generate data
    x_1 = jnp.array(make_moons(256, noise=0.05)[0])
    x_0 = jax.random.normal(prng_x_0, x_1.shape)
    t = jax.random.normal(prng_t, (len(x_1), 1))
    # sample point on the path
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    # calculate loss & update
    loss, grads = nnx.value_and_grad(loss_fn)(model, x_t, t, dx_t)
    optimizer.update(grads)
    return loss


def training():
    flow = Flow()
    model = flow

    rngs = nnx.Rngs(0)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2))
    # loss_fn = nn.MSELoss()

    pbar = tqdm(range(10000))
    for i in pbar:
        loss = train_step(model, optimizer, rngs())
        if i % 1000 == 0:
            pbar.set_description(f"loss = {loss}")



def sampling(flow):
    rngs = nnx.Rngs(113)
    x = jax.random.normal(rngs(), (300, 2))
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = jnp.linspace(0, 1.0, n_steps + 1)

    axes[0].scatter(x[:, 0], x[:, 1], s=10)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(n_steps):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.savefig("output/moons.jpg")