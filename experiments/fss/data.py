import time
import os
import pandas as pd
import httpx
import math
import logging
import urllib
import glob
import traceback

from tqdm import tqdm
from playwright.sync_api import sync_playwright
from experiments.fss.retry import retry


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DATA_DIR = "/data/fss"

# from: https://www.ifrs.org/-/media/feature/standards/taxonomy/2016/documentation-labels-in-excel/taxonomy-view-with-definitions-annual-2016.xlsx?la=en
IFRS_REF = f"{DATA_DIR}/ifrs-full.tsv"


def ifrs_concept_labels():
    ifrs = pd.read_csv(IFRS_REF, sep="\t")
    ifrs = ifrs[~pd.isna(ifrs.NAME)]
    mapping = dict(zip(ifrs["NAME"].values, ifrs["DOCUMENTATION LABEL"].values))
    return mapping



def download(urls: list[str], output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for url in tqdm(urls):
        logger.debug(f"Downloading: {url}")
        filename = urllib.parse.urlparse(url).path.lstrip("/").replace("/", "__")
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Already downloaded: {url}")
            continue
        resp = httpx.get(url)
        with open(filepath, "w") as fp:
            fp.write(resp.text)



def main():
    df = pd.read_parquet(f"{DATA_DIR}/xbrl-dump.parquet")

    # download all pages
    report_dir = os.path.join(DATA_DIR, "reports")
    urls = ["https://filings.xbrl.org" + url for url in df.report_url.unique()]
    download(urls, report_dir)

    # make screenshots
    screenshot_dir = os.path.join(DATA_DIR, "screenshots")
    report_paths = list(glob.glob(report_dir + "/*"))
    capture_with_retries = retry(times=3, exceptions=Exception)(capture_views)
    for ri, report_path in enumerate(report_paths):
        logger.info(f"Screenshot {ri+1}/{len(report_paths)}")
        base_name = os.path.splitext(os.path.basename(report_path))[0]
        report_screenshot_dir = os.path.join(screenshot_dir, base_name)
        try:
            capture_with_retries("file://" + report_path, report_screenshot_dir)
        except Exception:
            logger.warning(f"Failed to take screenshots of {report_path}")
            traceback.format_exc()




# async def main():
#     df = pd.read_parquet(f"{DATA_DIR}/xbrl-dump.parquet")
#     row = df.loc[5000]
#     resp = httpx.get("https://filings.xbrl.org" + row.report_url)
#     filepath = f"{DATA_DIR}/reports/{row.report_url.replace('/', '_')}"
#     with open(filepath, "w") as fp:
#         fp.write(resp.text)
#     xpath = row.element_xpath

#     await xpath_screenshot(filepath, meaningful_element_context(xpath))


# def meaningful_element_context(xpath: str):
#     parts = xpath.split("/")
#     split_point = [i for i in range(len(parts)) if parts[i].startswith("div")][-1] + 1
#     return "/".join(parts[:split_point])


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


# async def test():
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         page = await browser.new_page()
#         print("opening file")
#         await page.goto('file://' + filepath, wait_until="load")
#         await page.screenshot(
#             clip={"x": 0, "y": 10_000, "height": 1000, "width": 1000},
#             path="output/segment.png"
#         )


#         element = await page.query_selector("xpath=" + xpath)
#         # await element.is_visible()
#         await element.scroll_into_view_if_needed()
#         # page.wait_for_load_state("networkidle")
#         print("making a screenshot")
#         await page.screenshot(path='output/page.png', full_page=False, timeout=600_000)


def xpath_screenshot(filepath: str, xpath: str):
    # in our dataset, .element_xpath is specified without namespace,
    # but playwright requires it; thus we transform XPath in way that
    # ignores the xmlns attribute
    xpath = ignore_namespace(xpath)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("opening file")
        page.goto('file://' + filepath, wait_until="domcontentloaded")
        # page.wait_for_load_state("networkidle")
        # await page.screenshot(path='output/full_page.png', full_page=True, timeout=60000)
        print("loaded.")
        element = page.query_selector("xpath=" + xpath)
        element.is_visible()
        print("making screenshot")
        element.screenshot(path='output/element_screenshot.png', timeout=600_000)
        print(f"BOUNDING BOX: {element.bounding_box()}")
        print("closing")
        browser.close()



def capture_views(url: str, output_prefix: str, viewport_width=1280, viewport_height=1280):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        page = context.new_page()
        page.goto(url)

        # Get full page height
        page_height = page.evaluate("() => document.body.scrollHeight")
        num_views = math.ceil(page_height / viewport_height)

        for i in tqdm(range(num_views)):
            scroll_position = i * viewport_height
            page.evaluate(f"() => window.scrollTo(0, {scroll_position})")
            time.sleep(0.2)  # Allow scroll to settle/render
            path = f"{output_prefix}/{i:03d}.png"
            if not os.path.exists(path):
                page.screenshot(path=path)
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
