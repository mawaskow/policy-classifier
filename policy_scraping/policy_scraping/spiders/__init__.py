# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

import logging
import re
from scrapy.spiders import Spider
from scrapy.utils.log import configure_logging

class BaseSpider(Spider):
	logging.getLogger('scrapy.core.scraper').addFilter(lambda x: not x.getMessage().startswith('Scraped from')) #To avoid printing item in the terminal prompt
	configure_logging(install_root_handler=True)
	logging.disable(10)  # DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40; CRITICAL = 50