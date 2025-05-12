import jax
import jax.numpy as jnp
import flax.nnx as nnx


def pairwise_euclidean(x, y):
  assert x.ndim == y.ndim == 2
  return jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))


class Rounding:
    """
    from fabrique import LLM
    model_id = "microsoft/Phi-3.5-mini-instruct"
    llm = LLM.from_pretrained(model_id, max_seq_len=512, use_cache=False)
    embedding = llm.model.tok_embeddings

    rounding = Rounding(embedding)
    tokens = jnp.array(llm.tokenizer.encode("Nice weather").ids)
    x = embedding(tokens)
    indexes = rounding.top_k(x)[1]

    closest_tokens = llm.tokenizer.decode_batch(indexes)
    print(closest_tokens)
    """


    def __init__(self, embedding: nnx.Embed):
        self.embedding = embedding

    def top_k(self, x: jax.Array, k=5):
        """
        Find k most similar vectors to the given embedding matrix.

        Args:
            x (jax.Array): embedding array of shape (n_tokens, emb_size)
            k (int, default=5): number of top vectors to find

        Returns:
            values (jax.Array): Euclidian distance to the top tokens
            indexes (jax.Array): Indexes of top tokens
        """
        assert x.ndim == 2
        assert x.shape[-1] == self.embedding.embedding.shape[-1]
        adjacency = pairwise_euclidean(x, self.embedding.embedding)
        values, indexes = jax.lax.approx_min_k(adjacency, k=k)
        return values, indexes


def main():
    rngs = nnx.Rngs(0, params=483)
    # embedding = nnx.Embed(100, 10, rngs=rngs)
    # self = Rounding(embedding)
    # k = 5
    # x = jax.random.normal(rngs(), (4, 10))


    from fabrique import LLM
    model_id = "microsoft/Phi-3.5-mini-instruct"
    llm = LLM.from_pretrained(model_id, max_seq_len=512, use_cache=False)
    embedding = llm.model.tok_embeddings
    self = Rounding(embedding)
    k = 5
    # x = jax.random.normal(rngs(), (4, embedding.embedding.shape[-1]))

    tokens = jnp.array(llm.tokenizer.encode("Nice weather").ids)
    x = embedding(tokens)
    closest = self.top_k(x)[1]

    llm.tokenizer.decode_batch(closest)