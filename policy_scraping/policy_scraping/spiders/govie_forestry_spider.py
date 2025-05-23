import scrapy
import json
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import IrishGovPolicy
import hashlib
import os

# Global Variables
#
base_dir = os.getcwd()
kwds_forest = "\\keywords\\keywords_forestry.json"
kwds_anti = "\\keywords\\keywords_peat.json"
output_dir = "\\outputs\\forestry"
#
# Get files
#
with open(base_dir+kwds_forest, "r", encoding="utf-8") as infile:
    forkwddoc = json.load(infile)
with open(base_dir+kwds_anti, "r", encoding="utf-8") as infile:
    allkwddoc = json.load(infile)
    
sr_kw_dct = forkwddoc["srch_en"]
sr_akw_dct = allkwddoc["srch_anti_ie"]
doc_akw_dct = allkwddoc["doc_anti_ie"]

#
# spideytime
#
class GovIEForestSpider(BaseSpider):
    name = "goviefor"
    # in policy-classifier/policy_scraping/policy_scraping dir
    # scrapy crawl goviefor -O ../outputs/goviefor.json

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
        url = "https://www.gov.ie/en/policies/"
        yield scrapy.Request(url, self.parse_dept)

    def parse_dept(self, response):
        # points links to the publications tab of each department
        depts_of_int = ["268a7-agriculture-and-food/","d7a12b-climate-action-and-environment/","d5adb8-community-supports/", "b2a3c-food-vision-2030-a-world-leader-in-sustainable-food-systems/", "39e5f9-natural-resources/", "ac9ee6-action-plan-for-rural-development/"]
        dept_hrefs = ["https://www.gov.ie/en/policy/"+link+"latest/" for link in depts_of_int]
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
        try:
            page_next = response.selector.xpath('//a[@aria-label="next"]').attrib["href"]
            yield response.follow(page_next, self.nav_dept_pg)
        except:
            pass

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