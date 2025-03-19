import shutil
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from matplotlib import pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

from equilibrium.flow.generation import generate
from equilibrium.flow.path.affine import CondOTProbPath
from equilibrium.flow.path.path import ProbPath
from equilibrium.utils import load_model, plot_samples, save_model
from examples.text_onehot.encoder import OneHotEncoder, build_char_vocab
from examples.text_onehot.model import Transformer, ModelArgs


MODEL_PATH = "output/ckpt-text"
N_EPOCHS = 20


summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")


def loss_fn(
    model: nnx.Module,
    path: ProbPath,
    x: jax.Array,
    padding_mask: jax.Array,
    rngs: nnx.Rngs,
):
    x = x * 2.0 - 1.0
    noise = jax.random.normal(rngs.data(), shape=x.shape, dtype=x.dtype)
    t = jax.random.uniform(rngs.time(), x.shape[0], dtype=x.dtype)
    path_sample = path.sample(t=t, x_0=noise, x_1=x)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t
    loss = jnp.pow(model(x_t, padding_mask, t) - u_t, 2).mean()
    return loss


@nnx.jit(static_argnums=(1,))
def train_step(
    model: nnx.Module,
    path: ProbPath,
    x: jax.Array,
    padding_mask: jax.Array,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, path, x, padding_mask, rngs)
    optimizer.update(grads)
    return loss


def training():
    shutil.rmtree("output/ckpt-text")
    # shutil.rmtree("/tmp/tensorboard")

    rngs = nnx.Rngs(params=108, data=92, time=319)
    path = CondOTProbPath()
    bsz, max_length = 2, 32

    # load dataset
    ds = load_dataset("Abirate/english_quotes")["train"]
    batch = next(ds.iter(batch_size=bsz))
    # samples = jnp.array(batch["image"]).astype(jnp.bfloat16) / 255
    # labels = jnp.array(batch["label"])

    vocab = build_char_vocab(ds["quote"])
    encoder = OneHotEncoder(vocab, max_length=max_length)

    args = ModelArgs(dim=len(vocab), n_layers=8, n_heads=8, vocab_size=len(vocab), ffn_hidden_size=256)
    model = Transformer(args)

    # run train loop
    # optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=1e-3))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3))
    step = 0
    for epoch in range(N_EPOCHS):
        epoch_losses = []
        pbar = tqdm(ds.iter(batch_size=bsz), total=ds.shape[0] // bsz)
        for batch in pbar:
            try:
                x, padding_mask = encoder.encode(batch["quote"])
                loss = train_step(model, path, x, padding_mask, optimizer, rngs)
                epoch_losses.append(loss.item())
                pbar.set_description(f"epoch {epoch}: loss = {loss}")
                with summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=step)
                step += 1
            except:
                print(f"Failed to process batch #{pbar.n}")
        pbar.write(f"==> epoch {epoch}: avg loss = {jnp.array(epoch_losses).mean()}")
        save_model(model, MODEL_PATH + f"/{epoch}")


def sampling(model_path: str, encoder: OneHotEncoder, args: ModelArgs):
    import os
    with jax.checking_leaks():
        model_path = os.path.abspath("output/ckpt-text/9")
        args.max_seq_len = 32
        model = load_model(lambda: Transformer(args), model_path, to_cpu=True)

        rngs = nnx.Rngs(data=113)
        noise = jax.random.normal(rngs.data(), (1, 32, 128))
        padding_mask = jnp.ones(noise.shape[:2], dtype=bool)
        wrapper = lambda x, timesteps, extra: model(x, padding_mask, timesteps)
        y = generate(wrapper, noise, n_steps=8, method="midpoint")
        encoder.decode(y)
    # plot_samples(samples, "output/generated.jpg")
    # plt.clf()


def main():
    rngs = nnx.Rngs(0)
    ds = load_dataset("Abirate/english_quotes")["train"]
    vocab = build_char_vocab(ds["quote"])

    encoder = OneHotEncoder(vocab)

    args = ModelArgs(dim=len(vocab), n_layers=2, n_heads=8, vocab_size=len(vocab))
    model = Transformer(args)


    texts = ["Let's have a black celebration", "Let's come together!"]
    x, pad_mask = encoder.encode(texts)
    encoder.decode(x)
    t = jax.random.uniform(rngs(), x.shape[0])
    y = model(x, pad_mask, t)
    encoder.decode(y)

    top = y[-1, :, :].argsort(-1)
    for j in range(1, 6):
        ids = top[:, -j]
        letters = [encoder.id2token[id.item()] for id in ids]
        print("".join(letters))


    # next step: flow matching training

