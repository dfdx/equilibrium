import jax
import jax.numpy as jnp
from fabrique.models.common.embeddings import create_sinusoidal_positions
from fabrique.models.common.norm import RMSNorm
from flax import nnx
from flax.nnx.graph import Static
from datasets import load_dataset
import optax

from equilibrium.models.transformer import ModelArgs, TransformerBlock
from equilibrium.tokenizer import Tokenizer


# tokens -> Embed -> Encoder -> Decoder -> Classifier -> out_tokens


class Encoder(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0, noise=8)):
        self.args = args
        self.rngs = rngs
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.dim,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs,
        )

        sincos = Static(
            create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
        )

        self.layers = [
            TransformerBlock(args, sincos, rngs=rngs) for _ in range(args.n_layers)
        ]

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.linear_mean = nnx.Linear(args.dim, args.dim, use_bias=False, rngs=rngs)
        self.linear_std = nnx.Linear(args.dim, args.dim, use_bias=False, rngs=rngs)

    def __call__(self, tokens: jax.Array, padding_mask: jax.Array | None = None):
        h = self.tok_embeddings(tokens)
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                padding_mask=padding_mask,
            )
        h = self.norm(h)

        mean = self.linear_mean(h)
        std = jnp.exp(self.linear_std(h))

        key = self.rngs.noise()
        z = mean + std * jax.random.normal(key, mean.shape)
        return z, mean, std


class Decoder(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=10)):
        self.args = args
        self.rngs = rngs
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        sincos = Static(
            create_sinusoidal_positions(args.max_seq_len, args.dim // args.n_heads)
        )

        self.layers = [
            TransformerBlock(args, sincos, rngs=rngs) for _ in range(args.n_layers)
        ]

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nnx.Linear(args.dim, args.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, z: jax.Array, padding_mask: jax.Array | None = None):
        h = z
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                padding_mask=padding_mask,
            )
        h = self.norm(h)
        output = self.output(h).astype("float32")
        return output


class VAE(nnx.Module):
    def __init__(
        self, args: ModelArgs, *, rngs: nnx.Rngs = nnx.Rngs(params=792, noise=43)
    ):
        self.encoder = Encoder(args, rngs=rngs)
        self.decoder = Decoder(args, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        z, mean, std = self.encoder(tokens)
        logits = self.decoder(z)
        return logits, mean, std


def vae_loss(model: VAE, tokens: jax.Array):
    # TODO: take padding into account
    logits, mean, std = model(tokens)
    kl_loss = jnp.mean(
        0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
    )
    reconstruction_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
    return reconstruction_loss + 0.1 * kl_loss


@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, tokens: jax.Array):
    loss, grads = nnx.value_and_grad(vae_loss)(model, tokens)
    optimizer.update(grads)
    return loss



BATCH_SIZE = 8
TEXT_LEN = 128

def main():
    dataset = load_dataset("open-r1/codeforces")
    tokenizer = Tokenizer.from_pretrained("google/gemma-3-1b-it")

    args = ModelArgs(vocab_size=tokenizer.vocab_size, max_seq_len=TEXT_LEN)
    model = VAE(args)
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3))

    # tokens = jax.numpy.arange(32).reshape(-1, 1)
    batch = next(dataset["test"].iter(BATCH_SIZE))

    for epoch in range(100):
        for bi, batch in enumerate(dataset["test"].iter(batch_size=BATCH_SIZE)):
            texts = batch["description"]
            tokens, attn_mask = tokenizer(texts, max_length=TEXT_LEN, padding_length=TEXT_LEN)
            loss = train_step(model, optimizer, tokens)
            if bi % 10 == 0:
                print(f"Epoch {epoch} batch {bi} loss: {loss}")


    batch = next(dataset["test"].iter(BATCH_SIZE))
    texts = batch["description"]
    tokens, attn_mask = tokenizer(texts, max_length=TEXT_LEN, padding_length=TEXT_LEN)
    logits, _, _ = model(tokens)
    tokenizer.decode(logits.argmax(axis=-1))
