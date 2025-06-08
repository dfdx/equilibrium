import flax.nnx as nnx
import jax
import optax
import tensorflow as tf
from datasets import Dataset, load_dataset

from examples.text_onehot.encoder import OneHotEncoder, build_char_vocab
from examples.text_onehot.model import ModelArgs, Transformer

summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")


def loss_fn(model, batch: dict):
    emb = batch["emb"]
    inputs = emb[:, :-1, :]
    labels = emb[:, 1:].argmax(axis=-1)
    mask = batch["pad_mask"][:, 1:]
    logits = model(inputs, mask)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = (loss * mask).sum() / mask.sum()  # ignore loss at padding
    return loss, logits


@nnx.jit
def train_step(
    model, enc: jax.Array, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grad = grad_fn(model, enc)
    optimizer.update(grad)
    metrics.update(loss=loss)
    return loss


NUM_EPOCHS = 50
TOTAL_STEPS = 10_000
BATCH_SIZE = 8


def train(model, encoder: OneHotEncoder, ds: Dataset):
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    step = 0
    for epoch in range(NUM_EPOCHS):
        if step == TOTAL_STEPS:
            break
        metrics.reset()
        for i, orig_batch in enumerate(ds.iter(batch_size=BATCH_SIZE)):
            emb, pad_mask = encoder.encode(orig_batch["quote"])
            batch = {"emb": emb, "pad_mask": pad_mask}
            loss = train_step(model, batch, optimizer, metrics)
            print(
                f"Epoch {epoch}, step {step}: avg_loss = {metrics.compute()['loss']:.2f}; batch_loss = {loss:.2f}"
            )
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, metrics.compute()["loss"])
            step += 1
            if step == TOTAL_STEPS:
                print("Finished training!")
                break


def main():
    ds = load_dataset("Abirate/english_quotes")["train"]
    vocab = build_char_vocab(ds["quote"])

    encoder = OneHotEncoder(vocab)

    args = ModelArgs(dim=len(vocab), n_layers=2, n_heads=8, vocab_size=len(vocab))
    model = Transformer(args)

    train(model, encoder, ds)

    texts = ["Let's have a black celebration", "Let's come together!"]
    x, pad_mask = encoder.encode(texts)
    encoder.decode(x)
    y = model(x, pad_mask)
    encoder.decode(y)

    # next step: flow matching training
