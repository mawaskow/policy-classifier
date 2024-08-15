'''
So, I think I accidentally got my IP banned from the site so idk how I'm going to do this now.

###############
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
#from playwright.async_api import async_playwright
import hashlib
import os

# Global Variables
#
base_dir = os.getcwd()
keyword_file = "\\keywords\\keywords_peat.json"
output_dir = "\\outputs\\poland\\strategie"
#
# Get files
#
with open(base_dir+keyword_file, "r", encoding="utf-8") as infile:
    kwd_fl = json.load(infile)

sr_kw_dct = kwd_fl["srch_pl"]

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
            HEADERS = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                "Accept": "application/json,text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                }
            # GET request
            yield scrapy.Request("https://edziennik.minrol.gov.pl/search", self.parse, headers=HEADERS, meta={"playwright": True})
            # POST request
            '''
            yield scrapy.FormRequest(
                url="https://httpbin.org/post",
                formdata={"foo": "bar"},
                meta={"playwright": True},
            )
            '''

    def parse(self, response, **kwargs):
        print("\n\n\nyeah dawg\n\n\n")
        check = response.selector.xpath('//label[@class="control-label"]/text()').get()
        print(check)
        print(response.selector.xpath('//div'))
        # 'response' contains the page as seen by the browser
        yield {"url": response.url}