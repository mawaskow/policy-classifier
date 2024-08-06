import scrapy
from policy_scraping.spiders import BaseSpider
from policy_scraping.items import NECPs
import hashlib
import os

# Global Variables
#
base_dir = os.getcwd()
output_dir = "\\outputs"

#
# spideytime
#
class NECPSpider(BaseSpider):
    name = "necp"
    #scrapy crawl necp -O ../outputs/necp.json

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("ITEM_PIPELINES", {"scrapy.pipelines.files.FilesPipeline": 1}, priority="spider")
        settings.set("FILES_STORE", base_dir+output_dir, priority="spider")

    def start_requests(self):
        url = "https://commission.europa.eu/energy-climate-change-environment/implementation-eu-countries/energy-and-climate-governance-and-reporting/national-energy-and-climate-plans_en#national-energy-and-climate-plans-2021-2030"
        yield scrapy.Request(url, self.get_natns)

    def get_natns(self, response):
        # gets list of all countries NECPs
        #countries = response.selector.xpath('//div[@class="ecl-accordion"]//span[@class="ecl-accordion__toggle-title"]/text()').getall()
        dct = {}
        #for i in countries:
        #    dct[i] = {}
        
        # select all NECP country items
        results_sel = response.selector.xpath('//div[@class="ecl-accordion"]//div[@class="ecl-accordion__item"]')
        for resp in results_sel:
            cntry = resp.xpath('.//span[@class="ecl-accordion__toggle-title"]/text()').get()
            dct[cntry] = {}
            getli = resp.xpath('.//div[@class="ecl"]/ul/li[not(contains(text(),"Draft") or contains(text(),"Commission"))]')
            for li in getli:
                litxt = li.xpath('./text()').get()
                if litxt:
                    if "Final updated NECP" in litxt:
                        dct[cntry]["Final updated NECP"] = {}
                        ael = li.xpath('./a')
                        for lnk in ael:
                            name = lnk.xpath('./text()').get()
                            dct[cntry]["Final updated NECP"][name]= lnk.attrib['href']
                    else:
                        aes = li.xpath('./a[not(contains(text(),"Draft") or contains(text(),"Commission"))]')
                        for lnk in aes:
                            lnktxt = lnk.xpath('./text()').get()
                            if "Commission" in lnktxt:
                                pass
                            elif "factsheet" in lnktxt:
                                pass
                            elif "Staff Working Document" in lnktxt:
                                pass
                            else:
                                dct[cntry][lnktxt]= lnk.attrib['href']
        print(dct)
        yield dct