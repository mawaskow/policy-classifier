import scrapy
import json
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import IrishGovPolicy
import hashlib

# Global Variables
#
base_dir = "C:\\Users\\allie\\Documents\\GitHub\\policy-classifier\\policy_scraping"
keyword_file = "\\policy_scraping\\keywords_EN.json"
antikeyword_file = "\\policy_scraping\\antikeywords_EN.json"
output_dir = "\\outputs"
#
# Get files
#
with open(base_dir+keyword_file, "r", encoding="utf-8") as infile:
    kwdct = json.load(infile)
with open(base_dir+antikeyword_file, "r", encoding="utf-8") as infile:
    akwdct = json.load(infile)

#
# spideytime
#
class PolicySpider(BaseSpider):
    name = "irish"
    #scrapy crawl irish -O ../outputs/irish.json

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
        url = "https://www.gov.ie/en/policies/"
        yield scrapy.Request(url, self.parse_dept)

    def parse_dept(self, response):
        # gets links to departments from policies landing page
        results_sel = response.selector.xpath('//div[contains(@class, "reboot-content")]//ul')
        # points links to the publications tab of each department
        dept_hrefs = [link.attrib["href"]+"latest/" for link in results_sel.xpath('.//a')]
        search_lnks = []
        # points links to search results filtered by type and keyword
        for dept in dept_hrefs:
            for key in kwdct:
                search_lnks.append(dept+f"?q={key}&sort_by=published_date&type=policy_information&type=general_publications")
        yield from response.follow_all(search_lnks, self.nav_dept_pg)
    
    def nav_dept_pg(self, response):
        #results_count = response.selector.xpath('//div[@class="container margin-bottom-sm"]/div/div/p/strong/text()').get()
        results_sel = response.selector.xpath('//div[contains(@class, "reboot-content")]//ul')
        pol_links = [link.attrib["href"] for link in results_sel.xpath('.//a')]
        yield from response.follow_all(pol_links, self.parse)

        pg_count = response.selector.xpath('//div[@title="Pagination"]//div[contains(@class,"text-center")]/text()').get()
        if int(pg_count.split("/")[0]) < int(pg_count.split("/")[1]):
            page_next = response.selector.xpath('//div[@title="Pagination"]//a').attrib["href"]
            yield response.follow(page_next, self.nav_dept_pg)

    def parse(self, response):
        doc_itm = IrishGovPolicy()
        results_sel = response.selector.xpath('//div[@id="main"]/div/div/div/div')
        #
        doc_itm["title"] = results_sel.xpath(".//h1/text()").get()
        doc_itm["link"] = response.request.url
        doc_itm["publication_date"] = results_sel.xpath(".//p[text()[contains(.,'Published')]]/time").attrib["datetime"]
        doc_itm["department"] = results_sel.xpath(".//p[text()[contains(.,'From')]]/a/text()").get()
        doc_itm["type"] = results_sel.xpath(".//span/text()").get().strip()
        #
        results_pdfs = response.selector.xpath('//a[text()="Download"]')
        file_lst = []
        hash_dct = {}
        for result_pdf in results_pdfs:
            title = result_pdf.xpath('../../div/p[contains(@id,"download_title")]/text()').get()
            if not any(kwd in title for kwd in akwdct):
                path = result_pdf.xpath(".").attrib["href"]
                file_lst.append(path)
                hash = hashlib.sha1(path.encode()).hexdigest()
                hash_dct[hash] = path
        #
        doc_itm["file_urls"] = file_lst
        doc_itm["hash_name"] = hash_dct
        if file_lst:
            yield doc_itm