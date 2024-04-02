from pathlib import Path

import scrapy


from pathlib import Path

import scrapy


class PolicySpider(scrapy.Spider):
    name = "policies"
    start_urls = [
        "https://www.gov.ie/en/policies/",
        "https://www.gov.ie/en/policy/268a7-agriculture-and-food/latest/",
    ]

    # is this where these response parsers of the starting pages should occur?
    # or should there be another function
    # get count of categories [header links]
    ##category_ct = response.xpath('//strong/text()').get()
    ##category_ct = int(category_ct.split(" ")[0])
    # domain list links
    ##dom_lst = response.xpath('//a[contains(@title, "Policy")]').get()
    ##dom_links = {}
    ##print(type(dom_lst))
    ##for i in dom_lst:
    ##    dom_links[i.text] = i.get_attribute('href')
    # parsing 

    def parse(self, response):
        results_sel = response.selector.xpath('//div[contains(@class, "reboot-content")]//ul')
        for policy in results_sel.xpath('.//a'):
            result_p = policy.xpath('..//p/text()').get().split(";")
            yield {
                "title": policy.xpath(".").attrib["title"],
                "href": policy.xpath(".").attrib["href"],
                "date": result_p[0].strip(),
                "dept": result_p[1].strip(),
                "type": result_p[2].strip()
            }