# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class IrishGovPolicy(Item):
	title = Field()
	link = Field()
	publication_date = Field()
	department = Field()
	type = Field()
	file_urls = Field()
	hash_name = Field()
	pass
