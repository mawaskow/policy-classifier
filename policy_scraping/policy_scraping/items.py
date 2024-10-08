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

class PLMinRolPolicy(Item):
	hash_name = Field()
	doc_title = Field()
	pg_title = Field()
	pg_link = Field()
	file_urls = Field()
	publication_date = Field()
	department = Field()
	type = Field()

	pass

class NECPs(Item):
	hash_name = Field()
	title = Field()
	country = Field()
	language = Field()
	file_urls = Field()

	pass

class PLStrategie(Item):
	hash_name = Field()
	doc_title = Field()
	pg_title = Field()
	pg_link = Field()
	file_urls = Field()
	department = Field()

	pass

class XLSXPolicy(Item):
	name = Field()
	engname = Field()
	level = Field()
	country = Field()
	nuts1name = Field()
	nuts2name = Field()
	nuts3name = Field()
	localauth = Field()
	site = Field()
	classif = Field()
	dates = Field()
	publisher = Field()
	excerpt = Field()
	engexc = Field()
	abstract = Field()
	engabst = Field()
	lang = Field()
	link = Field()
	file_urls = Field()
	hash_name = Field()

	pass