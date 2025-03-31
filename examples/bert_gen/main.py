
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import tensorflow as tf
from datetime import datetime
from datasets import load_dataset, Dataset
from fabrique import LLM
from fabrique.loading import from_pretrained
from fabrique.models.bert.modeling import (
    ModelArgs as EncoderModelArgs,
    Transformer as Encoder
)
from examples.bert_gen.generator import (
    ModelArgs as GenModelArgs,
    Transformer as Generator,
    init_from
)
from examples.text_onehot.encoder import OneHotEncoder, build_char_vocab


RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M')
TENSORBOARD_PATH = f"/tmp/tensorboard/{RUN_TAG}"
MODEL_PATH = f"output/ckpt-text/{RUN_TAG}"
N_EPOCHS = 20


summary_writer = tf.summary.create_file_writer(TENSORBOARD_PATH)


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




# def main():
#     from equilibrium.utils.inspection import print_size

#     kw = {
#         "max_seq_len": 512, "max_batch_size": 1, "dtype": jnp.bfloat16, "param_dtype": jnp.bfloat16, "vocab_size": 32064, # "cond_dim": 768,
#     }
#     model = init_from("microsoft/Phi-3.5-mini-instruct", **kw)

#     from fabrique import LLM


#     llm = LLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", **kw)

#     # model = Transformer(ModelArgs(**kw), rngs=nnx.Rngs(54))
#     # model = nnx.eval_shape(lambda: Transformer(ModelArgs(**kw)))
#     _, state = nnx.split(llm.model)
#     model = partial_init(state, ModelArgs(**kw), nnx.Rngs(54))


def main():
    ds = load_dataset("Abirate/english_quotes")["train"]
    # vocab = build_char_vocab(ds["quote"])

    kw = {
        "max_seq_len": 512, "max_batch_size": 1, "dtype": jnp.bfloat16, "param_dtype": jnp.bfloat16, "vocab_size": 32064, # "cond_dim": 768,
    }
    with jax.checking_leaks():
        tokenizer, model = init_from("microsoft/Phi-3.5-mini-instruct", **kw)
        enc_tokenizer, encoder, _ = from_pretrained("google-bert/bert-base-uncased")

        cond_tokens = jnp.array([enc.ids for enc in enc_tokenizer.encode_batch(["Hello world"])])
        cond = encoder(cond_tokens)[:, 0, :]

        tokens = jnp.array([enc.ids for enc in enc_tokenizer.encode_batch(["<start>"])])
        model(tokens, start_pos=0, cond=cond)

    train(model, encoder, ds)


    texts = ["Let's have a black celebration", "Let's come together!"]
    x, pad_mask = encoder.encode(texts)
    encoder.decode(x)
    y = model(x, pad_mask)
    encoder.decode(y)

    # next step: flow matching training

