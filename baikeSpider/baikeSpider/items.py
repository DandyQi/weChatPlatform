# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class BaikeCategoryItem(scrapy.Item):
    category = scrapy.Field()
    url = scrapy.Field()
    childCategory = scrapy.Field()


class BaikeInstanceItem(scrapy.Item):
    category = scrapy.Field()
    instanceName = scrapy.Field()
    instanceAlias = scrapy.Field()
    instanceAbstract = scrapy.Field()
    instanceInternalLink = scrapy.Field()
    instanceInfobox = scrapy.Field()
    instanceContent = scrapy.Field()
    instanceTag = scrapy.Field()
