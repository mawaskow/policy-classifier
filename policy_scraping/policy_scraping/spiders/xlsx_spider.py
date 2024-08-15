import scrapy
import json
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import XLSXPolicy
import hashlib
import os
import pandas as pd

'''
Load in spreadsheet
extract list of pdfs
get pdfs
///
Load in spreadsheet
Parse into json
'''

# Global Variables
#
base_dir = os.getcwd()
output_dir = "\\outputs\\excel"
#
# Get files
#
XLSX_FILE = "C:/Users/allie/Documents/PhD/IrishPoliciesMar24/pdfsinpols.xlsx"
#

#
# spideytime
#
class XLSXSpider(BaseSpider):
    name = "excel"
    #scrapy crawl excel -O ../outputs/excel.json

    #df = pd.read_excel(XLSX_FILE)
    def __init__(self):
        self.df = pd.read_excel(XLSX_FILE)

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
        urls = list(self.df['longpdf_link'])
        for url in urls:
            try:
                yield scrapy.Request(url, self.parse)
            except Exception as e:
                print(f"Failed {url} due to {e}.")
                pass

    def parse(self, response):
        doc_itm = XLSXPolicy()
        doc_itm["name"] = str(response.headers["Content-Type"] )            
        doc_itm["file_urls"] = [response.request.url]
        hash = hashlib.sha1(response.request.url.encode()).hexdigest()
        doc_itm["hash_name"] = hash
        yield doc_itm