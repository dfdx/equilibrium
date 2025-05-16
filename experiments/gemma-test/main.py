import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
from gemma import gm
from gemma.multimodal.vision_utils import ViTModel
from experiments.fss.data import ifrs_concept_labels

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



SEARCH_PROMPT_PREFIX = """
You will be given a list of document pages and a question. Answer the question
and generate a JSON with the following fields:

* answer - answer to the question in concise form
* pages - on what page(s) you have found the answer
* texts - text(s) from that pages that you used for the answer
* comment - why you believe that this text(s) answer the question

If the answer is not in the text, set "answer" to null and keep other fields empty

"""


def find_answer(sampler: gm.text.ChatSampler, pages: list, question: str, step: int=5):
    for i in range(0, len(pages), step):
        print(f"Looking at pages {i}-{i + step - 1}")
        prompt = SEARCH_PROMPT_PREFIX
        start_page, end_page = i, min(i + step, len(pages))
        for i in range(start_page, end_page):
            prompt += f"Page {i} <start_of_image>\n"
        prompt += f"\n\nQuestion: {question}"

        out = sampler.chat(prompt, images=pages[start_page:end_page])
        answer = json.loads(out.strip("`").strip("json"))
        if answer["answer"] is not None:
            return answer
    return None




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

    page_dir = "/data/fss/screenshots/529900TYYSRJH2VJSP60__2024-12-31__ESEF__AT__0__529900TYYSRJH2VJSP60-2024-12-31-0-en__reports__529900TYYSRJH2VJSP60-2024-12-31-0-en"
    # img_path = img_dir + "/1.png"
    images = []
    img_paths = sorted(list(glob(page_dir + "/*.png")))
    for img_path in tqdm(img_paths):
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

    question = "What is The amount of income recognised from portfolio and other management fees?"

    img_from, img_to = 3, 5
    for i in range(img_from, img_to):
        prompt += f"Page {i} <start_of_image>\n"
    prompt += f"\n\nQuestion: {question}"

    out = sampler.chat(prompt, images=images[img_from:img_to])
    print(out)


    ifrs_map = ifrs_concept_labels()

# main_vqa()