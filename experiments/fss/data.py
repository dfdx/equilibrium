# # from selenium import webdriver
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import httpx

from playwright.async_api import async_playwright



DATA_DIR = "/data/fss"

async def main():
    df = pd.read_parquet(f"{DATA_DIR}/xbrl-dump.parquet")
    row = df.loc[5000]
    resp = httpx.get("https://filings.xbrl.org" + row.report_url)
    filepath = f"{DATA_DIR}/reports/{row.report_url.replace('/', '_')}"
    with open(filepath, "w") as fp:
        fp.write(resp.text)
    xpath = row.element_xpath

    await xpath_screenshot(filepath, meaningful_element_context(xpath))


def meaningful_element_context(xpath: str):
    parts = xpath.split("/")
    split_point = [i for i in range(len(parts)) if parts[i].startswith("div")][-1] + 1
    return "/".join(parts[:split_point])


def ignore_namespace(xpath: str):
    """
    Transform XPath in a way that ignores xmlns attribute.
    """
    parts = xpath.strip("/").split("/")
    new_parts = []
    for part in parts:
        tag = part.split("[", 1)[0]
        new_part = part.replace(tag, f"*[local-name()='{tag}']")
        new_parts.append(new_part)
    return "/" + "/".join(new_parts)


async def test():
    async with async_playwright() as p:
        # browser = await p.chromium.launch(headless=True)
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        print("opening file")
        await page.goto('file://' + filepath, wait_until="load")
        element = await page.query_selector("xpath=" + xpath)
        # await element.is_visible()
        await element.scroll_into_view_if_needed()
        # page.wait_for_load_state("networkidle")
        print("making a screenshot")
        await page.screenshot(path='output/page.png', full_page=False, timeout=600_000)


async def xpath_screenshot(filepath: str, xpath: str):
    # in our dataset, .element_xpath is specified without namespace,
    # but playwright requires it; thus we transform XPath in way that
    # ignores the xmlns attribute
    xpath = ignore_namespace(xpath)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("opening file")
        await page.goto('file://' + filepath, wait_until="domcontentloaded")
        # page.wait_for_load_state("networkidle")
        # await page.screenshot(path='output/full_page.png', full_page=True, timeout=60000)
        print("loaded.")
        element = await page.query_selector("xpath=" + xpath)
        await element.is_visible()
        print("making screenshot")
        await element.screenshot(path='output/element_screenshot.png', timeout=600_000)
        print(f"BOUNDING BOX: {await element.bounding_box()}")
        print("closing")
        await browser.close()


# def extract2():
#     filepath = '/data/fss/reports/_WOCMU6HCI0OJWNPRZS33_2024-12-31_ESEF_IT_0_WOCMU6HCI0OJWNPRZS33-2024-12-31-0-en_reports_WOCMU6HCI0OJWNPRZS33-2024-12-31-0-en.xhtml'
#     # xpath = '/html[1]/body[1]/div[400]'

#     xpath = '/html[1]/body[1]/div[400]/div[1]/div[1]/div[52]'
#     xpath = ignore_namespace(xpath)

#     from playwright.sync_api import sync_playwright

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
#         print("opening file")
#         page.goto('file://' + filepath, wait_until="domcontentloaded")
#         # page.wait_for_load_state("networkidle")
#         # page.screenshot(path='output/full_page.png', full_page=True, timeout=60000)
#         print("loaded.")
#         element = page.query_selector("xpath=" + xpath)
#         print(f"element is visible = {element.is_visible()}")
#         print("making screenshot")
#         element.screenshot(path='output/element_screenshot.png', timeout=600_000)
#         print("closing")
#         browser.close()



def gemma_test():
    # https://flax.readthedocs.io/en/latest/guides/gemma.html
    from gemma import gm

    # Model and parameters
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    # Example of multi-turn conversation
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        multi_turn=True,
    )

    prompt = """Which of the two images do you prefer?

    Image 1: <start_of_image>
    Image 2: <start_of_image>

    Write your answer as a poem."""
    out0 = sampler.chat(prompt, images=[image1, image2])

    out1 = sampler.chat('What about the other image ?')