def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_trunc_cosine(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "pw_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


alphas = 1.0 - betas
self.alphas_cumprod = np.cumprod(alphas, axis=0)
self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

# calculations for diffusion q(x_t | x_{t-1}) and others
self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)


def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data for a given number of diffusion steps.

    In other words, sample from q(x_t | x_0).

    :param x_start: the initial data batch.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :param noise: if specified, the split-out normal noise.
    :return: A noisy version of x_start.
    """
    if noise is None:
        noise = th.randn_like(x_start)
    assert noise.shape == x_start.shape
    return (
        _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        * noise
    )


x_start = self.get_x_start(x_start_mean, std)
# print(x_start_mean.shape, x_start.shape)
if noise is None:
    noise = th.randn_like(x_start)
x_t = self.q_sample(x_start, t, noise=noise)  # reparametrization trick.
