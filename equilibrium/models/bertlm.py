import jax
import jax.numpy as jnp
from flax import nnx

from fabrique.models.bert.modeling import (
    ModelArgs, Embeddings, TransformerBlock
)
from fabrique.loading import from_pretrained
from equilibrium.models.embeddings import timestep_embedding



class BertLM(nnx.Module):

    def __init__(self, args: ModelArgs, rngs: nnx.Rngs = nnx.Rngs(params=0)):
        """
        Initialize a Transformer model.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            embeddings (Embeddings): Embeddings module (positional + word + segment).
            layers (List[TransformerBlock]): List of transformer blocks.
            pooler (Pooler): Pooler module.
        """
        self.args = args
        self.embeddings = Embeddings(args, rngs=rngs)
        # self.time_embedding = TimeEmbedding(rngs=rngs)
        # self.time_adapter = nnx.Linear(self.time_embedding.dim * 4, args.dim, rngs=rngs)
        self.layers = [TransformerBlock(args, rngs=rngs) for _ in range(args.n_layers)]
        self.output = nnx.Linear(args.dim, args.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        tokens: jax.Array,
        mask: jax.Array | None = None,
        segments: jax.Array | None = None,
        timesteps: jax.Array | None = None,
        deterministic=True,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (jax.Array): Input token indices.
            mask (jax.Array): Attention mask.
            segments (jax.Array): Segment tokens.
            timesteps (jax.Array): Array of time steps, shape = (bs,).
            deterministic (bool): Deterministic flag for dropout.

        Returns:
            jax.Array: Output embeddings after applying the Transformer model.
        """
        if mask is None:
            mask = jnp.ones_like(tokens)
        if segments is None:
            segments = jnp.zeros_like(tokens)
        position_ids = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
        h = self.embeddings(tokens, segments, position_ids, deterministic=deterministic)
        # adding time embedding
        t_emb = timestep_embedding(timesteps, self.args.dim)
        # t_emb = self.time_adapter(t_emb)
        h = h + jnp.expand_dims(t_emb, 1)  # broadcast time emb to each token
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, deterministic=deterministic)
        output = self.output(h).astype("float32")
        return output

    @staticmethod
    def from_bert(model_id: str = "google-bert/bert-base-uncased", **kwargs):
        tokenizer, bert, hf_config = from_pretrained(model_id, **kwargs)
        model = BertLM(bert.args)
        model.embeddings = bert.embeddings
        model.layers = bert.layers
        model.output.kernel.value = model.embeddings.token_embeddings.embedding.value.T
        del bert
        return tokenizer, model, hf_config




from tokenizers import Tokenizer

def tokenize(tokenizer: Tokenizer, texts: list[str], pad_id: int = 0):
    # TODO: use mask for padding
    tokenizer.enable_padding(pad_id=pad_id)
    tokens = [e.ids for e in tokenizer.encode_batch(texts)]
    tokens = jnp.asarray(tokens)
    return tokens


def main():
    tokenizer, model, _ = BertLM.from_bert()
    # tokenizer, model, _ = from_pretrained("microsoft/Phi-3-mini-4k-instruct", max_seq_len=512, max_batch_size=2)
    texts = ["No man's sky", "Ça va?"]
    tokens = tokenize(tokenizer, texts)
    timesteps = jnp.asarray([3, 5])
    deterministic = True

    logits = model(tokens, timesteps=timesteps)
    out_tokens = logits.argmax(axis=-1)
    tokenizer.decode_batch(out_tokens)

