import math
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset, Dataset
from examples.text_onehot.model import ModelArgs, Transformer


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


summary_writer = tf.summary.create_file_writer("/tmp/tensorboard")


def closest_power_of_2(n: int, lower=True):
    log_n = math.log(n, 2)
    pow = math.floor(log_n) if lower else math.ceil(log_n)
    return 2**(pow)


def build_char_vocab(texts):
    chars = Counter()
    for quote in tqdm(texts):
        text = quote.strip().strip('”“')
        chars.update(text)
    target_vocab_size = closest_power_of_2(len(chars))
    char_list = [ch for ch, _ in chars.most_common(target_vocab_size - 2)]
    char_list.append(UNK_TOKEN)
    char_list.append(PAD_TOKEN)
    assert len(char_list) == target_vocab_size
    vocab = {ch: id for id, ch in enumerate(char_list)}
    return vocab


class OneHotEncoder:

    def __init__(self, vocab: dict, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, max_length: int = 512):
        self.token2id = vocab
        self.id2token = {id: token for token, id in vocab.items()}
        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]
        self.max_length = max_length

    def encode(self, texts: list[str]):
        length = max([len(text) for text in texts])
        length = closest_power_of_2(length, lower=False)
        length = min(length, self.max_length)
        bsz, length, vocab_size = len(texts), length, len(self.token2id)
        emb = np.zeros((bsz, length, vocab_size))
        pad_mask = np.ones((bsz, length), dtype=bool)
        for i, text in enumerate(texts):
            for j in range(length):
                if j < len(text):
                    ch = text[j]
                    k = self.token2id.get(ch, self.unk_id)
                else:
                    k = self.pad_id
                    pad_mask[i, j] = False
                emb[i, j, k] = 1
        return jnp.array(emb), jnp.array(pad_mask)

    def decode(self, emb: jax.Array):
        assert emb.ndim == 3, "Expected array of shape (bsz, seq_len, dim)"
        ids = emb.argmax(axis=-1)
        texts = []
        for tids in ids:
            chars = [self.id2token[tid.item()] for tid in tids if tid.item() != self.pad_id]
            text = "".join(chars)
            texts.append(text)
        return texts



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
def train_step(model, enc: jax.Array, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grad = grad_fn(model, enc)
    optimizer.update(grad)
    metrics.update(loss=loss)
    return loss


NUM_EPOCHS = 10
TOTAL_STEPS = 1000
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


