# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

#add location
#add governance level
#add search keyword that retrieved it?

class IrishGovPolicy(Item):
	hash_name = Field()
	doc_title = Field()
	pg_title = Field()
	pg_link = Field()
	file_urls = Field()
	publication_date = Field()
	department = Field()
	type = Field()

	pass