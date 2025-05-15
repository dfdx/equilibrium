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
import math
import asyncio

from tqdm import tqdm
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
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("opening file")
        await page.goto('file://' + filepath, wait_until="load")
        await page.screenshot(
            clip={"x": 0, "y": 10_000, "height": 1000, "width": 1000},
            path="output/segment.png"
        )


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



async def capture_views(url, output_prefix="output/screenshots", viewport_width=1280, viewport_height=1280):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        page = await context.new_page()
        await page.goto(url)

        # Get full page height
        page_height = await page.evaluate("() => document.body.scrollHeight")
        num_views = math.ceil(page_height / viewport_height)

        for i in tqdm(range(num_views)):
            scroll_position = i * viewport_height
            await page.evaluate(f"() => window.scrollTo(0, {scroll_position})")
            await asyncio.sleep(0.2)  # Allow scroll to settle/render

            # note: we can't create coroutines here and use gather at the end
            # because .screenshot() captures CURRENT view, which should NOT be moved
            # before screenshot is done
            await page.screenshot(path=f"{output_prefix}/{i+1}.png")
        browser.close()


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
