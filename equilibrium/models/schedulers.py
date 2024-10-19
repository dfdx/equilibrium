import jax
import jax.numpy as jnp


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, noise, timesteps):
        sqrt_alphas_cumprod_t = jnp.take(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)

        noisy_samples = (
            sqrt_alphas_cumprod_t[:, None, None] * x0 +
            sqrt_one_minus_alphas_cumprod_t[:, None, None] * noise
        )
        return noisy_samples

    def sample_x_t_prev(self, eps: jax.Array, x_t: jax.Array, timesteps: jax.Array, z: jax.Array):
        """
        Single inference step. Takes predicted noise (eps), current sample (x_t)
        and current timesteps, and returns sample at the previous timestep (x_{t-1}).

        Params:
        -------
        * eps: jax.Array
            Predicted noise
        * x_t: jax.Array
            Current sample at timestep t
        * timesteps: jax.Array
            Current timestep t
        * z: jax.Array
            Random noise to add to the sample. z ~ N(0, I)

        See details in Algorithm 2 (body) in https://arxiv.org/pdf/2006.11239
        or in Eq (87) in https://arxiv.org/pdf/2403.18103
        """
        t = timesteps
        # coefficients
        alphas_t = jnp.take(self.alphas, t)[:, None, None]    # perhaps only works for 3D inputs
        sqrt_one_minus_alphas_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, t)[:, None, None]
        betas_t = jnp.take(self.betas, t)[:, None, None]
        sigmas_t = jnp.sqrt(betas_t)   # according to Section 3.2 in https://arxiv.org/pdf/2006.11239
        # sampling
        mu_t = (1 - jnp.sqrt(alphas_t)) * (x_t - (1 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * eps)
        x_t_prev = mu_t + sigmas_t * z
        return x_t_prev

    def remove_noise(self, noisy_samples, noise, timesteps):
        sqrt_alphas_cumprod_t = jnp.take(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)

        denoised = (
            (noisy_samples - sqrt_one_minus_alphas_cumprod_t[:, None, None] * noise) /
            sqrt_alphas_cumprod_t[:, None, None]
        )
        return denoised

    def get_sampling_timesteps(self, num_inference_steps=None):
        num_inference_steps = num_inference_steps or self.num_train_timesteps
        return jnp.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).astype(jnp.int32)




def main():
    eps = jax.random.normal(jax.random.key(0), (8, 128, 768))
    x_t = jax.random.normal(jax.random.key(1), (8, 128, 768))
    z = jax.random.normal(jax.random.key(2), x_t.shape)

    timesteps = jnp.arange(8) * 10
    self = DDPMScheduler()