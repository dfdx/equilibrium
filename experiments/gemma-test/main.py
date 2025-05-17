import jax
import jax.numpy as jnp
# import tensorflow_datasets as tfds
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from gemma import gm
from experiments.fss.data import ifrs_concept_labels, DATA_DIR, url_to_dirname

# def main():
#     ds = tfds.data_source('oxford_flowers102', split='train')
#     image1 = ds[0]["image"]
#     image2 = ds[1]["image"]

#     # Model and parameters
#     model = gm.nn.Gemma3_4B()
#     params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

#     # Example of multi-turn conversation
#     sampler = gm.text.ChatSampler(
#         model=model,
#         params=params,
#         multi_turn=True,
#     )

#     prompt = """What do you see on the image?

#     Image: <start_of_image>

#     Write your answer as a poem."""
#     out0 = sampler.chat(prompt, images=[image1])

#     out1 = sampler.chat('What about the other image ?')


#     ve = ViTModel()
#     fresh = ve.init(jax.random.key(1), image1[None, :, :, :])["params"]
#     saved = params["vision_encoder"]["siglip_encoder"]

#     r = ve.apply({"params": fresh}, image1[None, :, :, :])



def load_page(img_path: str):
    return jnp.asarray(Image.open(img_path).resize((896, 896)))


SEARCH_PROMPT_PREFIX = """
You are an IFRS expert. You will be given a list of document pages and a concept label
from IFRS. Your task is to determine if any of these pages discloses information
from the IFRS label.

If there's a value associated with the label, extract it. For example, if the label
asks about total company's income in the reporting period, and this income is explicitly
mentioned in a text or table, then return the income value. If there's no associated value
or the information in question is not disclosed, return `null` as value.

You answer should be in JSON format with the following fields:

* disclosed (boolean) - whether the information from IFRS label is disclosed
* value (str) - the associated value, if any
* pages - on what page(s) you have found the answer
* texts - text(s) from that pages that you used for the answer
* comment - why you believe that this text(s) answer the question

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
        value = answer.get("value")
        if value is not None and value != "null":
            return answer
    return None




def main_vqa():
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        cache_length=4096,
        # multi_turn=True,
    )

    # page_dir = "/data/fss/screenshots/529900TYYSRJH2VJSP60__2024-12-31__ESEF__AT__0__529900TYYSRJH2VJSP60-2024-12-31-0-en__reports__529900TYYSRJH2VJSP60-2024-12-31-0-en"
    # # img_path = img_dir + "/1.png"
    # pages = []
    # img_paths = sorted(list(glob(page_dir + "/*.png")))
    # for img_path in tqdm(img_paths):
    #     img = jnp.asarray(Image.open(img_path).resize((896, 896)))
    #     pages.append(img)


    ifrs_map = ifrs_concept_labels()
    # questions = list(ifrs_map.values())

    # question = f"Is this information disclosed? {questions[12]}"

    # print(question)
    # find_answer(sampler, pages, question, step=3)


    df = pd.read_parquet(f"{DATA_DIR}/xbrl-dump.parquet")

    # inferred_report_url = "/" + page_dir.lstrip("/data/fss/screenshots").replace("__", "/") + ".xhtml"
    ifrs_df = df[(df.standard == "IFRS")]
    row = ifrs_df.iloc[100_000]
    row.report_url
    label = ifrs_map[row.concept.lstrip("ifrs-full:")]

    question = f"IFRS label: {label}"
    page_dir = f"{DATA_DIR}/screenshots/{url_to_dirname(row.report_url)}"
    page_paths = sorted(list(glob(page_dir + "/*.png")))
    pages = [load_page(path) for path in tqdm(page_paths)]
    ans = find_answer(sampler, pages, question, step=3)

    row.value
