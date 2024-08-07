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
        # gets links to departments from policies landing page
        print("\n\n\nyeah dawg\n\n\n")
        results_sel = response.selector.xpath('//div[contains(@class, "reboot-content")]//ul')
        # points links to the publications tab of each department
        dept_hrefs = [link.attrib["href"]+"latest/" for link in results_sel.xpath('.//a')]
        search_lnks = []
        # points links to search results filtered by type and keyword
        for dept in dept_hrefs:
            for key in sr_kw_dct:
                search_lnks.append(dept+f"?q={key}&sort_by=published_date&type=policy_information&type=general_publications")
        yield from response.follow_all(search_lnks, self.nav_dept_pg)
    
    def nav_dept_pg(self, response):
        # get links to all results on current search page
        results_sel = response.selector.xpath('//div[contains(@class, "reboot-content")]//ul')
        pol_links = [link.attrib["href"] for link in results_sel.xpath('.//a')]
        yield from response.follow_all(pol_links, self.parse)
        # if there is another page, go to the next page so the above can be executed again
        pg_count = response.selector.xpath('//div[@title="Pagination"]//div[contains(@class,"text-center")]/text()').get()
        if int(pg_count.split("/")[0]) < int(pg_count.split("/")[1]):
            page_next = response.selector.xpath('//div[@title="Pagination"]//a').attrib["href"]
            yield response.follow(page_next, self.nav_dept_pg)

    def parse(self, response):
        # on the page of the result entry
        results_sel = response.selector.xpath('//div[@id="main"]/div/div/div/div')
        pg_title = results_sel.xpath(".//h1/text()").get()
        # if the search antikeywords arent present
        if not any(akwd in pg_title.lower() for akwd in sr_akw_dct):
            # get all possible download docs on page
            results_pdfs = response.selector.xpath('//a[text()="Download"]')
            for result_pdf in results_pdfs:
                doc_title = result_pdf.xpath('../../div/p[contains(@id,"download_title")]/text()').get()
                if not any(akwd in doc_title.lower() for akwd in doc_akw_dct):
                    doc_itm = IrishGovPolicy()
                    doc_itm["pg_title"] = pg_title
                    doc_itm["pg_link"] = response.request.url
                    doc_itm["publication_date"] = results_sel.xpath(".//p[text()[contains(.,'Published')]]/time").attrib["datetime"]
                    doc_itm["department"] = results_sel.xpath(".//p[text()[contains(.,'From')]]/a/text()").get()
                    doc_itm["type"] = results_sel.xpath(".//span/text()").get().strip()
                    #
                    doc_itm["doc_title"] = doc_title
                    path = result_pdf.xpath(".").attrib["href"]
                    doc_itm["file_urls"] = [path]
                    hash = hashlib.sha1(path.encode()).hexdigest()
                    doc_itm["hash_name"] = hash
                    yield doc_itm