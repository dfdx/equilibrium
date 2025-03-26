import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from tqdm import tqdm


def midpoint_step(model, x_t, t_start, t_end):
    t_start = t_start.reshape(-1)
    t_end = t_end.reshape(-1)
    dt = t_end - t_start
    # Step 1: Compute the slope at the beginning of the interval
    k1 = model(x=x_t, timesteps=t_start.reshape(1), extra={"labels": jnp.array([0])})
    # Step 2: Estimate the state at the midpoint
    x_mid = x_t + (dt / 2) * k1
    # Step 3: Compute the slope at the midpoint
    k2 = model(x=x_mid, timesteps=t_start + dt / 2, extra={"labels": jnp.array([0])})
    # Step 4: Update the state using the midpoint slope
    x_next = x_t + dt * k2
    return x_next


def repeat_time(model):
    def wrapper(y, t):
        ta = jnp.repeat(t, y.shape[0], axis=0)
        return model(y, ta, {})

    return wrapper


def generate(model, noise: jax.Array, n_steps: int = 8, method: str = "midpoint"):
    time_steps = jnp.linspace(0, 1.0, n_steps + 1)
    if method == "midpoint":
        x = noise
        samples = []
        for i in tqdm(range(n_steps)):
            x = midpoint_step(
                model, x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1]
            )
            samples.append(x)
        samples = jnp.vstack(samples)
        return samples
    elif method == "odeint":
        wrapper = repeat_time(model)
        samples = odeint(wrapper, noise, time_steps)
        samples = samples.squeeze(axis=1)
        return samples
    else:
        raise ValueError(
            f"Unsupported generation method: {method}. "
            + "Supported methods are: midpoint | odeint"
        )
