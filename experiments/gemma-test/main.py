import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from glob import glob
from tqdm import tqdm
from PIL import Image
from gemma import gm
from gemma.multimodal.vision_utils import ViTModel


def main():
    ds = tfds.data_source('oxford_flowers102', split='train')
    image1 = ds[0]["image"]
    image2 = ds[1]["image"]

    # Model and parameters
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    # Example of multi-turn conversation
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        multi_turn=True,
    )

    prompt = """What do you see on the image?

    Image: <start_of_image>

    Write your answer as a poem."""
    out0 = sampler.chat(prompt, images=[image1])

    out1 = sampler.chat('What about the other image ?')


    ve = ViTModel()
    fresh = ve.init(jax.random.key(1), image1[None, :, :, :])["params"]
    saved = params["vision_encoder"]["siglip_encoder"]

    r = ve.apply({"params": fresh}, image1[None, :, :, :])




def main_vqa():
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    # Example of multi-turn conversation
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        cache_length=4096,
        # multi_turn=True,
    )

    img_dir = "output/screenshots"
    # img_path = img_dir + "/1.png"
    images = []
    for img_path in tqdm(glob(img_dir + "/*.png")):
        img = jnp.asarray(Image.open(img_path).resize((896, 896)))
        images.append(img)

    prompt = """
    You will be given a list of document pages and a question. Answer the question
    and generate a JSON with the following fields:

    * answer - answer to the question
    * pages - on what page(s) you have found the answer
    * texts - text(s) from that pages that you used for the answer
    * comment - why you believe that this text(s) answer the question

    """
    img_from, img_to = 23, 27
    for i in range(img_from, img_to):
        prompt += f"Page {i} <start_of_image>\n"
    prompt += "\n\nQuestion: How many public charging points does the company have?"

    out = sampler.chat(prompt, images=images[img_from:img_to])
    print(out)

# main_vqa()