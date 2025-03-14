import os
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx
from tqdm import tqdm

from equilibrium.unet import UNet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Prevent TFDS from using GPU
tf.config.experimental.set_visible_devices([], "GPU")


BATCH_SIZE = 8
IMG_SIZE = (64, 64)


def get_datasets():
    # Load the MNIST dataset
    dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    def load_image(datapoint):
        input_image = tf.image.resize(datapoint["image"], IMG_SIZE)
        input_mask = tf.image.resize(
            datapoint["segmentation_mask"],
            IMG_SIZE,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        input_image, input_mask = normalize(input_image, input_mask)
        return input_image, input_mask

    train_images = dataset["train"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return train_images


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def loss_fn(model: UNet, batch: Tuple[jax.Array, jax.Array], timestamps: jax.Array):
    logits = model(batch[0], timestamps)
    mask = jnp.clip(batch[1], min=0, max=1)
    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=mask).mean()
    return loss, logits


@nnx.jit
def train_step(model: UNet, optimizer: nnx.Optimizer, batch):
    """Train for a single step."""
    # timestamps aren't really used for segmentation, so filling it with dummy values
    timestamps = jnp.zeros(batch[0].shape[0])
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch, timestamps)
    optimizer.update(grads)
    return loss


def main():
    train_images = get_datasets()
    train_batches = (
        train_images.cache().batch(BATCH_SIZE)
        # .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])

    plt.savefig("output/output.png")

    model = UNet(dim=32, in_channels=3, out_channels=1)

    batch = next(train_batches.as_numpy_iterator())
    batch = [jnp.asarray(u) for u in batch]
    x, mask = batch
    timestamps = jax.random.randint(
        jax.random.key(0), (BATCH_SIZE,), minval=0, maxval=100
    )
    y = model(x, timestamps)

    loss, logits = loss_fn(model, batch, timestamps)

    learning_rate = 0.005
    momentum = 0.9

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))

    num_epochs = 30
    num_steps_per_epoch = train_batches.cardinality().numpy() // num_epochs

    for epoch in range(num_epochs):
        epoch_losses = []
        for step, batch in tqdm(enumerate(train_batches.as_numpy_iterator())):
            loss = train_step(model, optimizer, batch)
            epoch_losses.append(loss.item())
            # print(f"Epoch {epoch}, step {step}, loss = {loss.item()}")
        print(f">>> Epoch {epoch} mean loss = {jnp.asarray(epoch_losses).mean()}")
        images, masks = batch
        pred_masks = optimizer.model(images, jnp.zeros(images.shape[0]))
        display([images[0], masks[0], pred_masks[0]])
        plt.savefig(f"output/predicted_epoch_{epoch}.png")
