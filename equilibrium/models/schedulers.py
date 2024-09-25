import jax.numpy as jnp


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alphas_cumprod_t = jnp.take(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas_cumprod_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)

        noisy_samples = (
            sqrt_alphas_cumprod_t[:, None, None] * original_samples +
            sqrt_one_minus_alphas_cumprod_t[:, None, None] * noise
        )
        return noisy_samples

    # def remove_noise(self, noisy_samples, noise, timesteps):
    #     alpha_t = jnp.take(self.alphas_cumprod, timesteps)
    #     sqrt_one_minus_alpha_t = jnp.take(self.sqrt_one_minus_alphas_cumprod, timesteps)

    #     denoised = (
    #         (noisy_samples - sqrt_one_minus_alpha_t[:, None, None] * noise) /
    #         jnp.sqrt(alpha_t)[:, None, None]
    #     )
    #     return denoised

    # def get_sampling_timesteps(self, num_inference_steps):
    #     return jnp.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).astype(jnp.int32)


