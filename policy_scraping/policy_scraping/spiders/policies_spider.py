from pathlib import Path
import scrapy
import json
from policy_scraping.spiders import BaseSpider

with open("C:\\Users\\allie\\Documents\\GitHub\\policy-classifier\\policy_scraping\\policy_scraping\\keywords_EN.json", "r", encoding="utf-8") as infile:
    kwdct = json.load(infile)

class PolicySpider(BaseSpider):
    name = "irish"

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
        results_sel = response.selector.xpath('//div[@id="main"]/div/div/div/div')
        print(results_sel.xpath(".//h1/text()").get())
        print(results_sel.xpath(".//p[contains(@text(), 'Published')]/time/text()").get())
        print(results_sel.xpath(".//p[contains(@text(), 'From')]/a/text()").get())
        print(results_sel.xpath(".//span/text()").get().strip())
        yield {
            "title": results_sel.xpath(".//h1/text()").get(),
            "date": results_sel.xpath(".//p[contains(@text(), 'Published')]/time/text()").get(),
            "dept": results_sel.xpath(".//p[contains(@text(), 'From')]/a/text()").get(),
            "type": results_sel.xpath(".//span/text()").get().strip(),
        }