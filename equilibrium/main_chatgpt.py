import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import math
from flax.training import train_state

# Sinusoidal Time Embedding
class SinusoidalPositionalEmbedding(nn.Module):
    embedding_dim: int
    max_len: int = 5000

    def __call__(self, timesteps):
        position = jnp.arange(self.max_len).reshape(-1, 1).astype(jnp.float32)
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2).astype(jnp.float32) *
                           (-math.log(10000.0) / self.embedding_dim))
        pe = jnp.zeros((self.max_len, self.embedding_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe[timesteps]

# Transformer Denoiser
class TransformerDenoiser(nn.Module):
    embedding_dim: int
    num_heads: int
    num_layers: int
    vocab_size: int
    dropout: float = 0.1

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.embedding_dim)
        self.time_embedding = SinusoidalPositionalEmbedding(self.embedding_dim)
        self.transformer_layers = [
            nn.SelfAttention(self.embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ]
        self.output_layer = nn.Dense(self.vocab_size)

    def __call__(self, x, timesteps, train: bool = True):
        time_emb = self.time_embedding(timesteps)
        x = self.token_embedding(x) + time_emb
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# Diffusion Model
class DiffusionModel(nn.Module):
    denoiser: TransformerDenoiser
    num_timesteps: int = 1000

    def setup(self):
        self.beta = jnp.linspace(0.0001, 0.02, self.num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jnp.cumprod(self.alpha)

    def forward_diffusion(self, x0, t):
        noise = jax.random.normal(jax.random.PRNGKey(0), x0.shape)
        sqrt_alpha_bar_t = jnp.sqrt(self.alpha_bar[t]).reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = jnp.sqrt(1.0 - self.alpha_bar[t]).reshape(-1, 1)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def p_sample(self, xt, t, params):
        predicted_noise = self.denoiser.apply(params, xt, t)
        alpha_t = self.alpha[t].reshape(-1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)
        sqrt_one_minus_alpha_t = jnp.sqrt(1.0 - alpha_t).reshape(-1, 1)
        sqrt_alpha_bar_t = jnp.sqrt(alpha_bar_t).reshape(-1, 1)
        pred_x0 = (xt - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_bar_t
        return pred_x0

    def generate(self, seq_len, params):
        xt = jax.random.normal(jax.random.PRNGKey(0), (1, seq_len))
        for t in reversed(range(self.num_timesteps)):
            xt = self.p_sample(xt, jnp.array([t]), params)
        return xt


def main():
    embedding_dim = 128
    num_heads = 8
    num_layers = 6
    seq_len = 20
    vocab_size = 5000
    num_timesteps = 1000

    # Initialize model and optimizer
    denoiser = TransformerDenoiser(embedding_dim, num_heads, num_layers, vocab_size)
    variables = denoiser.init(jax.random.key(0), jnp.ones((1, seq_len)), jnp.array([0]))

    diffusion_model = DiffusionModel(denoiser)

    params = diffusion_model.init(jax.random.PRNGKey(0), jnp.ones((1, seq_len)), jnp.array([0]))
    tx = optax.adam(learning_rate=1e-4)
    state = train_state.TrainState.create(apply_fn=diffusion_model.apply, params=params, tx=tx)

    # Forward diffusion example
    x0 = jax.random.randint(jax.random.PRNGKey(1), (1, seq_len), 0, vocab_size)
    t = jax.random.randint(jax.random.PRNGKey(2), (1,), 0, num_timesteps)
    xt, noise = diffusion_model.forward_diffusion(x0, t)

    # Generation example
    generated_sample = diffusion_model.generate(seq_len, state.params)
