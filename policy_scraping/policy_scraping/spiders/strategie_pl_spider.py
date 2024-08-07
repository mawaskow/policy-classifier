import scrapy
import json
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import PLStrategie
import hashlib
import os

# Global Variables
#
base_dir = os.getcwd()
output_dir = "\\outputs\\poland\\strategie"
keyword_file = "\\keywords\\keywords_peat.json"

with open(base_dir+keyword_file, "r", encoding="utf-8") as infile:
    kwd_fl = json.load(infile)

doc_akw_dct = kwd_fl["doc_anti_pl"]
#
# spideytime
#
class PlStratSpider(BaseSpider):
    name = "plstrat"
    #scrapy crawl plstrat -O ../outputs/plstrat.json
    #scrapy shell -s USER_AGENT='Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' 'https://bip.mos.gov.pl/strategie-plany-programy/'

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
        url = "https://bip.mos.gov.pl/strategie-plany-programy/"
        #headers = {"User-Agent":'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0'}
        yield scrapy.Request(url, self.find_strats)

    def find_strats(self, response):
        # gets links to strategies, plans, and programs (spp)
        #print("\n\n\nyeah dawg 1\n\n\n")
        results_sel = response.selector.xpath('//div[contains(@class,"menu-block")]/a')
        # points links to the pages of each spp
        spp_hrefs = [link.attrib["href"] for link in results_sel]
        page_lnks = ["https://bip.mos.gov.pl/"+link for link in spp_hrefs]
        yield from response.follow_all(page_lnks, self.parse)

    def parse(self, response):
        #print("\n\n\nyeah dawg 2\n\n\n")
        # on the page of the spp
        pg_title = response.selector.xpath('//h1[@class="first-headline"]/text()').get()
        # get all possible download docs on page
        results_pdfs = response.selector.xpath('//div[contains(@class,"media-list")]/p[@class="media-heading"]/a')
        for result_pdf in results_pdfs:
            doc_title = result_pdf.xpath('./span/text()').get()
            if not any(akwd in doc_title.lower() for akwd in doc_akw_dct):
                doc_itm = PLStrategie()
                doc_itm["pg_title"] = pg_title
                doc_itm["pg_link"] = response.request.url
                doc_itm["department"] = "Ministerstwo Klimatu i Środowiska"
                #
                doc_itm["doc_title"] = doc_title
                path = "https://bip.mos.gov.pl/"+result_pdf.attrib["href"]
                doc_itm["file_urls"] = [path]
                hash = hashlib.sha1(path.encode()).hexdigest()
                doc_itm["hash_name"] = hash
                yield doc_itm