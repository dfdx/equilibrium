from functools import partial
import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer

from tqdm import tqdm

from equilibrium.models.bertlm import BertLM, tokenize
from equilibrium.models.schedulers import DDPMScheduler



def get_dataset():
    ds = load_dataset("billsum", split="ca_test")
    ds = ds.with_format("jax")
    ds = ds.remove_columns(["text", "summary"]).rename_column("title", "text")
    return ds


def diffusion_loss_fn(model, noise_scheduler, x):
    # Generate random noise
    noise = jax.random.normal(jax.random.key(0), x.shape)
    # Sample random timesteps
    timesteps = jax.random.randint(jax.random.key(0), (x.shape[0],), 0, noise_scheduler.num_train_timesteps)
    # Add noise to the input
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    # Predict the noise
    predicted_noise = model(noisy_x, timesteps=timesteps)
    # Compute MSE between true and predicted noise
    loss = jnp.mean((noise - predicted_noise) ** 2)
    x_rec = noise_scheduler.remove_noise(noisy_x, noise, timesteps)
    return loss, x_rec


@partial(nnx.jit, static_argnums=[1])
def loss_fn(model, noise_scheduler, tokens: jax.Array):
    x = model.embed(tokens)
    diff_loss, x_rec = diffusion_loss_fn(model, noise_scheduler, x)
    logits = model.get_logits(x_rec)
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens).mean()
    return diff_loss + ce_loss


def train(model, noise_scheduler, data: Dataset, tokenizer: Tokenizer, num_epochs: int = 10, batch_size: int = 8, learning_rate=1e-5):
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    losses = []
    for epoch in range(num_epochs):
        for texts in data.iter(batch_size=batch_size):
            tokens = tokenize(tokenizer, texts["text"])
            loss, grads = nnx.value_and_grad(loss_fn)(model, noise_scheduler, tokens)
            optimizer.update(grads)
            losses.append(loss.item())
            print(f"current loss = {loss.item()}")

        print(f"Epoch {epoch}, Loss: {jnp.asarray(losses).mean()}")

    return model


def sample(model, noise_scheduler, shape = (8, 128, 768), rng: nnx.Rngs = nnx.Rngs(0)):
    # Start from pure noise
    x = jax.random.normal(rng(), shape)

    # Iteratively denoise
    for t in tqdm(noise_scheduler.get_sampling_timesteps()):
        timesteps = jnp.full((shape[0],), t)
        eps = model(x, timesteps=timesteps)
        z = jax.random.normal(rng(), x.shape)
        # z = jnp.zeros(x.shape)
        x = noise_scheduler.sample_x_t_prev(eps, x, timesteps, z)
    return x


def main():
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_explain_cache_misses", True)


    tokenizer, model, _ = BertLM.from_bert()
    noise_scheduler = DDPMScheduler()
    data = get_dataset()
    model = train(model, noise_scheduler, data, tokenizer, num_epochs=10)


    x_gen = sample(model, noise_scheduler)
    tokens_gen = model.get_tokens(x_gen)
    tokenizer.decode_batch(tokens_gen)

    # noise reconstruction check
    texts = next(data.iter(batch_size=8))["text"]
    tokens = tokenize(tokenizer, texts)
    x = model.embed(tokens)
    noise = jax.random.normal(jax.random.key(0), (8, 128, 768))
    timesteps = jnp.asarray([500] * 8)
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
    denoised_x = noise_scheduler.remove_noise(noisy_x, noise, timesteps)
    x - denoised_x

    # diffusion_model = DiffusionModel(model, noise_scheduler)
    # self = diffusion_model

    tokens = tokenize(tokenizer, ["Hello, world!", "I am an olive"])
    x = model.embed(tokens_gen)


    num_epochs: int = 10
    batch_size: int = 8
    learning_rate = 1e-5

    texts = next(iter(data.iter(batch_size=batch_size)))




def main_sharding():
    from typing import Optional
    import numpy as np
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    mesh = Mesh(devices=mesh_utils.create_device_mesh((2, 1, 1)),
                axis_names=("b", "s", "d"))
    x = jax.device_put(x, NamedSharding(mesh, P("b", "s", "d")))





# class DiffusionModel(nnx.Module):

#     def __init__(self, bert_lm: BertLM, noise_scheduler):
#         self.model = bert_lm
#         self.noise_scheduler = noise_scheduler

#     def __call__(self, x, timesteps, deterministic=True):
#         return self.model(x, timesteps=timesteps, deterministic=deterministic)

#     def add_noise(self, original_samples, noise, timesteps):
#         return self.noise_scheduler.add_noise(original_samples, noise, timesteps)

#     # def remove_noise(self, noisy_samples, timesteps):
#     #     # Predict the noise component
#     #     predicted_noise = self(noisy_samples, timesteps)

#     #     # Use the noise scheduler to remove the predicted noise
#     #     denoised = self.noise_scheduler.remove_noise(noisy_samples, predicted_noise, timesteps)

#     #     return denoised

#     def sample(self, shape, num_inference_steps):
#         # Start from pure noise
#         x = jax.random.normal(nnx.next_rng_key(), shape)

#         # Iteratively denoise
#         for t in self.noise_scheduler.get_sampling_timesteps(num_inference_steps):
#             timesteps = jnp.full((shape[0],), t)
#             x = self.remove_noise(x, timesteps)

#         return x

#     # def train_step(self, x):
#     #     # Generate random noise
#     #     noise = jax.random.normal(jax.random.key(0), x.shape)

#     #     # Sample random timesteps
#     #     timesteps = jax.random.randint(jax.random.key(0), (x.shape[0],), 0, self.noise_scheduler.num_train_timesteps)

#     #     # Add noise to the input
#     #     noisy_x = self.add_noise(x, noise, timesteps)

#     #     # Predict the noise
#     #     predicted_noise = self.model(noisy_x, timesteps=timesteps)

#     #     # Compute loss (e.g., MSE between true and predicted noise)
#     #     loss = jnp.mean((noise - predicted_noise) ** 2)

#     #     return loss

#     def train(self, train_data, num_epochs, batch_size, learning_rate):
#         optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate))

#         @nnx.jit
#         def train_step(state, batch):
#             def loss_fn(params):
#                 self.update(params)
#                 return self.train_step(batch)

#             loss, grads = jax.value_and_grad(loss_fn)(state.params)
#             state = state.apply_gradients(grads=grads)
#             return state, loss

#         state = nnx.TrainState.create(
#             apply_fn=self.apply,
#             params=self.parameters(),
#             tx=optimizer
#         )

#         for epoch in range(num_epochs):
#             for batch in data_loader(train_data, batch_size):
#                 state, loss = train_step(state, batch)

#             print(f"Epoch {epoch}, Loss: {loss}")

#         self.update(state.params)

#     @classmethod
#     def from_pretrained(cls, model_id, noise_scheduler, **kwargs):
#         tokenizer, bert_lm, hf_config = BertLM.from_bert(model_id, **kwargs)
#         return cls(bert_lm, noise_scheduler), tokenizer, hf_config
