from typing import Tuple

import jax
import optax
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random as r
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
import jax.random as random
import orbax.checkpoint
import flax.nnx as nnx
from flax.training import train_state
import matplotlib.pyplot as plt

from typing import Callable
from tqdm import tqdm
from PIL import Image

from equilibrium.unet import UNet


# Prevent TFDS from using GPU
tf.config.experimental.set_visible_devices([], 'GPU')

# Defining some hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_STEPS_PER_EPOCH = 60000//BATCH_SIZE # MNIST has 60,000 training samples


# Load MNIST dataset

def get_datasets():
  # Load the MNIST dataset
    train_ds = tfds.load('mnist', as_supervised=True, split="train")

    # Normalization helper
    def preprocess(x, y):
        return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

    # Normalize to [-1, 1], shuffle and batch
    train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Return numpy arrays instead of TF tensors while iterating
    return tfds.as_numpy(train_ds)



# Defining a constant value for T
timesteps = 200

# Defining beta for all t's in T steps
beta = jnp.linspace(0.0001, 0.02, timesteps)

# Defining alpha and its derivatives according to reparameterization trick
alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha, 0)
alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = jnp.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)


# Implement noising logic according to reparameterization trick
def forward_noising(key, x_0, t):
    noise = random.normal(key, x_0.shape)
    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(jnp.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise



def loss_fn(model: UNet, x: jax.Array, noise: jax.Array, timestamps: jax.Array):
    pred_noise = model(x, timestamps)
    loss = jnp.mean((noise - pred_noise) ** 2)   # MSE
    return loss


@nnx.jit
def train_step(model: UNet, optimizer: nnx.Optimizer, x: jax.Array, noise: jax.Array, timestamps: jax.Array):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x, noise, timestamps)
    optimizer.update(grads)
    return loss



def train_epoch(epoch_num: int, model: UNet, optimizer: nnx.Optimizer, train_ds, rng):
    epoch_loss = []
    for index, images in enumerate(tqdm(train_ds)):
        # Creating two keys: one for timestamp generation and second for generating the noise
        rng, tsrng = random.split(rng)   # TODO: return rng?

        # Generating timestamps for this batch
        timestamps = random.randint(tsrng,
                                    shape=(images.shape[0],),
                                    minval=0, maxval=timesteps)

        # Generating the noise and noisy image for this batch
        noisy_images, noise = forward_noising(rng, images, timestamps)

        loss = train_step(model, optimizer, noisy_images, noise, timestamps)

        # Loss logging
        epoch_loss.append(loss.item())
        if index % 100 == 0:
            print(f"Loss at step {index}: ", loss)

        # Timestamps are not needed anymore. Saves some memory.
        del timestamps
    train_loss = np.mean(epoch_loss)
    return train_loss, rng




def train_model():
    train_ds = get_datasets()
    model = UNet(dim=32, in_channels=1, out_channels=1)

    x = next(iter(train_ds))
    timestamps = jax.random.randint(jax.random.key(0), (BATCH_SIZE,), minval=0, maxval=100)
    y = model(x, timestamps)

    noise = jnp.zeros_like(x)
    loss = loss_fn(model, x, noise, timestamps)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.005))
    num_epochs = 10
    rng = jax.random.key(0)

    for epoch in range(num_epochs):
        loss, rng = train_epoch(epoch, model, optimizer, train_ds, rng)
        print(f"====== Epoch {epoch}, loss: {loss} ======")
        save_model(model, os.path.abspath(f"output/unet_checkpoint_{epoch}"))

    return model



def save_model(model: UNet, path: str):
    state = nnx.state(model)
    # Save the parameters
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(f'{path}/state', state)


def load_model(path: str, *args, **kwargs) -> UNet:
    # create that model with abstract shapes
    # model = nnx.eval_shape(lambda: UNet(*args, **kwargs))
    model = UNet(*args, **kwargs)
    state = nnx.state(model)
    # Load the parameters
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore(f'{path}/state', item=state)
    # update the model with the loaded state
    nnx.update(model, state)
    return model



# This function defines the logic of getting x_t-1 given x_t
def backward_denoising_ddpm(x_t, pred_noise, t):
    alpha_t = jnp.take(alpha, t)
    alpha_t_bar = jnp.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = 1 / (alpha_t ** 0.5) * (x_t - eps_coef * pred_noise)

    var = jnp.take(beta, t)
    z = random.normal(key=random.PRNGKey(r.randint(1, 100)), shape=x_t.shape)

    return mean + (var ** 0.5) * z



# Save a GIF using logged images

def save_gif(img_list, path=""):
    # Transform images from [-1,1] to [0, 255]
    imgs = (Image.fromarray(np.array((np.array(i) * 127.5) + 1, np.int32)) for i in img_list)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)



def main():
    path = os.path.abspath("output/unet_checkpoint_7")
    model = load_model(path, dim=32, in_channels=1, out_channels=1)
    x = random.normal(random.PRNGKey(42), (1, 32, 32, 1))

    # Create a list to store output images
    img_list_ddpm = []

    # Append the initial noise to the list of images
    img_list_ddpm.append(jnp.squeeze(jnp.squeeze(x, 0),-1))

    # Iterate over T timesteps
    for i in tqdm(range(0, timesteps - 1)):
        # t-th timestep
        t = jnp.expand_dims(jnp.array(timesteps - i - 1, jnp.int32), 0)

        # Predict noise using U-Net
        pred_noise = model(x, t)

        # Obtain the output from the noise using the formula seen before
        x = backward_denoising_ddpm(x, pred_noise, t)

        # Log the image after every 25 iterations
        if i % 25 == 0:
            img_list_ddpm.append(jnp.squeeze(jnp.squeeze(x, 0),-1))
            plt.imshow(jnp.squeeze(jnp.squeeze(x, 0),-1), cmap='gray')
            plt.show()

    # Display the final generated image
    plt.imshow(jnp.squeeze(jnp.squeeze(x, 0),-1), cmap='gray')
    plt.show()

    # Save generated GIF
    save_gif(img_list_ddpm, path="output/output_ddpm.gif")
