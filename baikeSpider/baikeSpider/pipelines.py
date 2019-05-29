# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


import codecs
from scrapy.exporters import JsonItemExporter
from baikeSpider.items import BaikeCategoryItem, BaikeInstanceItem


class BaikePipeline(object):
    def __init__(self):
        self.file = codecs.open('baidu_category.csv', 'wb', encoding='utf-8')

        self.instance_file = open('baidu_instance.json', 'wb')
        self.exporter = JsonItemExporter(self.instance_file, encoding='utf-8', ensure_ascii=False)
        self.exporter.start_exporting()

    def process_item(self, item, spider):
        if isinstance(item, BaikeCategoryItem):
            item = dict(item)
            self.file.write('%s\t%s\t%s\n'
                            % (item["category"], item["url"], item["childCategory"]))
            return item

        if isinstance(item, BaikeInstanceItem):
            self.exporter.export_item(item)
            return item

    def close_spider(self, spider):
        self.file.close()

        self.exporter.finish_exporting()
        self.instance_file.close()
