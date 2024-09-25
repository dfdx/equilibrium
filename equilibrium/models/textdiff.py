import jax
import optax
import jax.numpy as jnp
import flax.nnx as nnx

from datasets import load_dataset, Dataset

from equilibrium.models.bertlm import BertLM, tokenize
from equilibrium.models.schedulers import DDPMScheduler



def get_dataset():
    ds = load_dataset("billsum", split="ca_test")
    ds = ds.with_format("jax")
    ds = ds.remove_columns(["text", "summary"]).rename_column("title", "text")
    return ds


def loss_fn(model, noise_scheduler, x):
    # Generate random noise
    noise = jax.random.normal(jax.random.key(0), x.shape)

    # Sample random timesteps
    timesteps = jax.random.randint(jax.random.key(0), (x.shape[0],), 0, noise_scheduler.num_train_timesteps)

    # Add noise to the input
    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

    # Predict the noise
    predicted_noise = model(noisy_x, timesteps=timesteps)

    # Compute loss (e.g., MSE between true and predicted noise)
    loss = jnp.mean((noise - predicted_noise) ** 2)

    return loss


def train(model, noise_scheduler, data: Dataset, num_epochs: int = 1, batch_size: int = 8, learning_rate=1e-5):
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    # @nnx.jit
    # def train_step(state, batch):
    #     def loss_fn(params):
    #         self.update(params)
    #         return self.train_step(batch)

    #     loss, grads = jax.value_and_grad(loss_fn)(state.params)
    #     state = state.apply_gradients(grads=grads)
    #     return state, loss

    # state = nnx.TrainState.create(
    #     apply_fn=self.apply,
    #     params=self.parameters(),
    #     tx=optimizer
    # )

    losses = []
    for epoch in range(num_epochs):
        for texts in data.iter(batch_size=batch_size):
            tokens = tokenize(tokenizer, texts["text"])
            x = model.embed(tokens)
            loss, grads = nnx.value_and_grad(loss_fn)(model, noise_scheduler, x)
            optimizer.update(grads)
            losses.append(loss.item())
            print(f"current loss = {loss.item()}")

        print(f"Epoch {epoch}, Loss: {jnp.asarray(losses).mean()}")



def main():
    tokenizer, model, _ = BertLM.from_bert()
    noise_scheduler = DDPMScheduler()
    diffusion_model = DiffusionModel(model, noise_scheduler)
    self = diffusion_model

    tokens = tokenize(tokenizer, ["Hello, world!", "I am an olive"])
    x = model.embed(tokens)

    data = get_dataset()
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate = 1e-5

    texts = next(iter(data.iter(batch_size=batch_size)))["text"]



    x_hat = model(x, timesteps=timesteps)
    out_tokens = model.get_tokens(x_hat)

    tokenizer.decode_batch(out_tokens)



    diffusion_model.train(train_data, num_epochs, batch_size, learning_rate)





class DiffusionModel(nnx.Module):

    def __init__(self, bert_lm: BertLM, noise_scheduler):
        self.model = bert_lm
        self.noise_scheduler = noise_scheduler

    def __call__(self, x, timesteps, deterministic=True):
        return self.model(x, timesteps=timesteps, deterministic=deterministic)

    def add_noise(self, original_samples, noise, timesteps):
        return self.noise_scheduler.add_noise(original_samples, noise, timesteps)

    # def remove_noise(self, noisy_samples, timesteps):
    #     # Predict the noise component
    #     predicted_noise = self(noisy_samples, timesteps)

    #     # Use the noise scheduler to remove the predicted noise
    #     denoised = self.noise_scheduler.remove_noise(noisy_samples, predicted_noise, timesteps)

    #     return denoised

    def sample(self, shape, num_inference_steps):
        # Start from pure noise
        x = jax.random.normal(nnx.next_rng_key(), shape)

        # Iteratively denoise
        for t in self.noise_scheduler.get_sampling_timesteps(num_inference_steps):
            timesteps = jnp.full((shape[0],), t)
            x = self.remove_noise(x, timesteps)

        return x

    # def train_step(self, x):
    #     # Generate random noise
    #     noise = jax.random.normal(jax.random.key(0), x.shape)

    #     # Sample random timesteps
    #     timesteps = jax.random.randint(jax.random.key(0), (x.shape[0],), 0, self.noise_scheduler.num_train_timesteps)

    #     # Add noise to the input
    #     noisy_x = self.add_noise(x, noise, timesteps)

    #     # Predict the noise
    #     predicted_noise = self.model(noisy_x, timesteps=timesteps)

    #     # Compute loss (e.g., MSE between true and predicted noise)
    #     loss = jnp.mean((noise - predicted_noise) ** 2)

    #     return loss

    def train(self, train_data, num_epochs, batch_size, learning_rate):
        optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate))

        @nnx.jit
        def train_step(state, batch):
            def loss_fn(params):
                self.update(params)
                return self.train_step(batch)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        state = nnx.TrainState.create(
            apply_fn=self.apply,
            params=self.parameters(),
            tx=optimizer
        )

        for epoch in range(num_epochs):
            for batch in data_loader(train_data, batch_size):
                state, loss = train_step(state, batch)

            print(f"Epoch {epoch}, Loss: {loss}")

        self.update(state.params)

    @classmethod
    def from_pretrained(cls, model_id, noise_scheduler, **kwargs):
        tokenizer, bert_lm, hf_config = BertLM.from_bert(model_id, **kwargs)
        return cls(bert_lm, noise_scheduler), tokenizer, hf_config
