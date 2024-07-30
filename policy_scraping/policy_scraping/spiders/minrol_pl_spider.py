'''
The Windows implementation of asyncio can use two event loop implementations: SelectorEventLoop, default before Python 3.8, required when using Twisted. ProactorEventLoop, default since Python 3.8, cannot work with Twisted.

So on Python 3.8+ the event loop class needs to be changed.

Changed in version 2.6.0: The event loop class is changed automatically when you change the TWISTED_REACTOR setting or call install_reactor().

To change the event loop class manually, call the following code before installing the reactor:

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
You can put this in the same function that installs the reactor, if you do that yourself, or in some code that runs before the reactor is installed, e.g. settings.py.
'''

import scrapy
import json
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import PLMinRolPolicy
from playwright.async_api import async_playwright
import hashlib
import os

# Global Variables
#
#base_dir = "C:\\Users\\ales\\Documents\\GitHub\\policy-classifier\\policy_scraping"
base_dir = os.getcwd()
keyword_file = "\\keywords\\keywords_peat.json"
output_dir = "\\outputs\\poland"
#
# Get files
#
with open(base_dir+keyword_file, "r", encoding="utf-8") as infile:
    kwd_fl = json.load(infile)

sr_kw_dct = kwd_fl["srch_pl"]
#sr_akw_dct = kwd_fl["srch_anti_ie"]
#doc_akw_dct = kwd_fl["doc_anti_ie"]

#
# spideytime
#
'''
class MinRolPLSpider(BaseSpider):
    name = "minrolpl"
    #scrapy crawl minrolpl -O ../outputs/minrolpl.json

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    start_urls = ["data:,"]  # avoid using the default Scrapy downloader

    async def parse(self, response):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.goto("https://edziennik.minrol.gov.pl/search")
            title = await page.title()
            return {"title": title}
'''
class MinRolPLSpider(BaseSpider):
    name = "minrolpl"
    #scrapy crawl minrolpl -O ../outputs/minrolpl.json

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
            # GET request
            yield scrapy.Request("https://httpbin.org/get", meta={"playwright": True})
            # POST request
            yield scrapy.FormRequest(
                url="https://httpbin.org/post",
                formdata={"foo": "bar"},
                meta={"playwright": True},
            )

    def parse(self, response, **kwargs):
        # 'response' contains the page as seen by the browser
        return {"url": response.url}