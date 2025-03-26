import datasets
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from matplotlib import pyplot as plt
from tqdm import tqdm

from equilibrium.flow.generation import generate
from equilibrium.flow.path.affine import CondOTProbPath
from equilibrium.flow.path.path import ProbPath
from equilibrium.models.unet import UNetModel
from equilibrium.utils import load_model, plot_samples, save_model

MODEL_PATH = "output/ckpt"
N_EPOCHS = 10


def loss_fn(
    model: nnx.Module,
    path: ProbPath,
    samples: jax.Array,
    labels: jax.Array,
    rngs: nnx.Rngs,
):
    samples = samples * 2.0 - 1.0
    noise = jax.random.normal(rngs(), shape=samples.shape, dtype=samples.dtype)
    t = jax.random.uniform(rngs(), samples.shape[0], dtype=samples.dtype)
    path_sample = path.sample(t=t, x_0=noise, x_1=samples)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t
    loss = jnp.pow(model(x_t, t, extra={"labels": labels}) - u_t, 2).mean()
    return loss


@nnx.jit(static_argnums=(1,))
def train_step(
    model: nnx.Module,
    path: ProbPath,
    samples: jax.Array,
    labels: jax.Array,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, path, samples, labels, rngs)
    optimizer.update(grads)
    return loss


def training():
    m_rngs = nnx.Rngs(0)  # model
    t_rngs = nnx.Rngs(108)  # data
    path = CondOTProbPath()
    bsz, in_channels, hw = 2, 3, 64
    model = UNetModel(in_channels, rngs=m_rngs)

    # load dataset
    ds = datasets.load_dataset("Maysee/tiny-imagenet", split="train[:10%]")
    batch = next(ds.iter(batch_size=bsz))
    samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
    labels = jnp.array(batch["label"])

    # run train loop
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=1e-3))
    for epoch in range(N_EPOCHS):
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
        pbar.write(f"==> epoch {epoch}: avg loss = {jnp.array(epoch_losses).mean()}")
        save_model(model, MODEL_PATH + f"/{epoch}")


def sampling(model_path: str):
    model_path = "/data/equilibrium/unet/49/"
    model = load_model(lambda: UNetModel(3, rngs=nnx.Rngs(28)), model_path, to_cpu=True)
    rngs = nnx.Rngs(2)
    noise = jax.random.normal(rngs(), (1, 64, 64, 3))
    samples = generate(model, noise, n_steps=8, method="midpoint")
    plot_samples(samples, "output/generated.jpg")
    plt.clf()
