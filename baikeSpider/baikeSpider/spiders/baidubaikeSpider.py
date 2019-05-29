# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/2/27 16:00
# File usage:

import re
import json
import scrapy
import urllib
import pandas as pd
from baikeSpider.items import BaikeCategoryItem, BaikeInstanceItem


class BaidubaikeCategory(scrapy.Spider):
    name = 'baiduCategory'

    def start_requests(self):
        urls = [
            'http://baike.baidu.com/fenlei/%E7%BB%8F%E6%B5%8E'
        ]

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # parse category and its child categories
        category = urllib.unquote(response.url.split('/')[-1]).decode('utf-8')
        url = response.url

        child_category = []
        for child in response\
                .xpath('//div[not(contains(@class, "brother-list"))]/div[@class="category-title "]/a/text()'):
            child_category.append(child.extract())
        if len(child_category) != 0:
            child_category = ";".join(child_category)
        else:
            child_category = "No_child"

        yield BaikeCategoryItem(category=category, url=url, childCategory=child_category)

        # parse the child categories in the same way
        for child_url in response\
                .xpath('//div[not(contains(@class, "brother-list"))]/div[@class="category-title "]/a/@href').extract():
            child_url = 'http://baike.baidu.com' + child_url
            yield scrapy.Request(url=child_url, callback=self.parse)


class BaidubaikeInstance(scrapy.Spider):
    name = 'baiduInstance'
    tag_id_set = set()

    def start_requests(self):
        # categories = pd.read_csv('baidu_category.csv',
        #                          sep='\t', names=['category', 'url', 'childCategory'], encoding='utf-8')
        # for index, row in categories.iterrows():
        #     yield scrapy.Request(url=row['url'], callback=self.parse, meta={'category': row['category']})

        # test_url = 'https://baike.baidu.com/item/%E5%A7%94%E6%AF%94'
        # yield scrapy.Request(url=test_url, callback=self.parse_detail, meta={'category': '', 'alias': ''})
        tags = pd.read_csv('baikeSpider/tagid', sep='\t', names=['tagName', 'tagId'], encoding='utf-8', dtype=str)
        for index, row in tags.iterrows():
            url = 'https://baike.baidu.com/wikitag/api/getlemmas'
            for p in range(1, 20):
                form_data = {
                    'contentLength': '40',
                    'filterTags': '%5B%5D',
                    'fromLemma': 'false',
                    'limit': '100',
                    'page': str(p),
                    'tagId': row["tagId"],
                    'timeout': '3000'
                }
                yield scrapy.FormRequest(url, callback=self.parse_wikitag_detail,
                                         formdata=form_data, meta={"category": row["tagName"]})

    def parse(self, response):
        # parse the instances in this category
        instance_url = response.xpath('//div[@class="grid-list grid-list-spot"]/ul/li/div[@class="list"]/a/@href')\
            .extract()
        if instance_url:
            instance_alias = response\
                .xpath('//div[@class="grid-list grid-list-spot"]/ul/li/div[@class="list"]/a/text()').extract()
            instance_detail = zip(instance_url, instance_alias)
            for detail in instance_detail:
                url = 'http://baike.baidu.com' + detail[0]
                yield scrapy.Request(url=url, callback=self.parse_detail,
                                     meta={
                                         'alias': detail[1],
                                         'category': response.meta['category'],
                                     })

        next_page_url = response.xpath('//div[@class="page"]/a[@id="next"]/@href').extract()
        if next_page_url:
            next_page_url = 'http://baike.baidu.com/fenlei/' + next_page_url[0]
            yield scrapy.Request(url=next_page_url, callback=self.parse, meta={'category': response.meta['category']})

    def parse_detail(self, response):
        category = response.meta['category']
        alias = response.meta['alias']
        instance_name = response.xpath('//dd[@class="lemmaWgt-lemmaTitle-title"]/h1/text()').extract()
        if instance_name:
            instance_name = instance_name[0]
            instance_abstract = response.xpath('//div[@class="lemma-summary"]').xpath('string(.)').extract()
            if instance_abstract:
                instance_abstract = "".join(instance_abstract)
                sup_notation = response.xpath('//div[@class="lemma-summary"]/div/sup').xpath('string(.)').extract()
                if sup_notation:
                    for notation in sup_notation:
                        instance_abstract = instance_abstract.replace(notation, '').strip()
            else:
                instance_abstract = None

            instance_internal_link_name = response.xpath('//div[@class="para"]/a/text()').extract()
            if instance_internal_link_name:
                instance_internal_link_url = response.xpath('//div[@class="para"]/a/@href').extract()
                instance_internal_link = dict((key, value) for key, value in
                                              zip(instance_internal_link_name, instance_internal_link_url))
            else:
                instance_internal_link = None

            instance_infobox_name = response.xpath('//dt[@class="basicInfo-item name"]').xpath('string(.)').extract()
            if instance_infobox_name:
                instance_infobox_value = response.xpath('//dd[@class="basicInfo-item value"]') \
                    .xpath('string(.)').extract()
                instance_infobox = dict((key.strip(), value.strip()) for key, value in
                                        zip(instance_infobox_name, instance_infobox_value))
            else:
                instance_infobox = None

            instance_content = "".join(response.xpath('//div[@class="para"]').xpath('string(.)').extract()) \
                .replace('\n', '')
            if instance_content == "":
                instance_content = None

            tag1 = response.xpath('//dd[@id="open-tag-item"]/span/a/text()').extract()
            tag1 = [item.strip() for item in tag1 if item.strip() != ""]
            tag2 = response.xpath('//dd[@id="open-tag-item"]/span/text()').extract()
            tag2 = [item.strip() for item in tag2 if item.strip() != ""]

            instance_tags = tag1 + tag2

            instance_tags_link = response.xpath('//dd[@id="open-tag-item"]/span/a/@href').extract()
            instance_tags_id = [re.findall(r"tagId=(\d+)", link)[0] for link in instance_tags_link]
            instance_tags_id = instance_tags_id + [""] * (len(instance_tags) - len(instance_tags_id))

            instance_tags = dict((key, value) for key, value in zip(instance_tags, instance_tags_id))

            yield BaikeInstanceItem(
                category=category,
                instanceName=instance_name,
                instanceAlias=alias,
                instanceAbstract=instance_abstract,
                instanceInternalLink=instance_internal_link,
                instanceInfobox=instance_infobox,
                instanceContent=instance_content,
                instanceTag=instance_tags
            )

    def parse_wikitag_detail(self, response):
        json_res = json.loads(response.body)
        item_list = json_res["lemmaList"]
        for item in item_list:
            url = item["lemmaUrl"]
            yield scrapy.Request(url=url, callback=self.parse_detail,
                                 meta={
                                     'alias': item["lemmaTitle"],
                                     'category': response.meta["category"],
                                 })
