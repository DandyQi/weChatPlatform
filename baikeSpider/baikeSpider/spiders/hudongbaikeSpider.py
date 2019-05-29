# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/3/5 15:12
# File usage:

import json
import scrapy
import urllib
import pandas as pd
from baikeSpider.items import BaikeCategoryItem, BaikeInstanceItem


class HudongbaikeCategory(scrapy.Spider):
    name = 'hudongSpider'

    def start_requests(self):
        # urls = [
        #     'http://fenlei.baike.com/%E7%BB%8F%E6%B5%8E'
        # ]
        #
        # for url in urls:
        #     yield scrapy.Request(url=url, callback=self.parse)

        # 我这里是限定了一些我需要的类别爬取，所以有一个本地文件，你可以选择全部都爬下来，就不需要这个本地文件
        df = pd.read_csv('hudong_category_filter.csv', sep='\t', encoding='utf-8')
        for idx, row in df.iterrows():
            category = row['category']
            instance_url = 'http://fenlei.baike.com/categorySpecialTopicAction.do?action=showDocInfo'
            for p in range(1, 20):
                form_data = {
                    'categoryName': category,
                    'pagePerNum': '100',
                    'pageNow': str(p)
                }
                yield scrapy.FormRequest(instance_url, callback=self.parse_api_result,
                                         formdata=form_data, meta={'category': category})

    # def parse(self, response):
    #     category = urllib.unquote(response.url.split('/')[-1]).decode('utf-8')
    #
    #     if len(response.xpath('//div[@class="sort_all up"]/p')) > 1:
    #         child = response.xpath('//div[@class="sort_all up"]/p')[1]
    #         child_categories = ';'.join(child.xpath('a/text()').extract())
    #     elif len(response.xpath('//div[@class="sort"]/p')) > 1:
    #         child = response.xpath('//div[@class="sort"]/p')[1]
    #         child_categories = ';'.join(child.xpath('a/text()').extract())
    #     else:
    #         child_categories = 'No_child'
    #
    #     yield BaikeCategoryItem(category=category, url=response.url, childCategory=child_categories)
    #
    #     # instance_url = 'http://fenlei.baike.com/categorySpecialTopicAction.do?action=showDocInfo'
    #     # for p in range(1, 20):
    #     #     form_data = {
    #     #         'categoryName': category,
    #     #         'pagePerNum': '100',
    #     #         'pageNow': str(p)
    #     #     }
    #     #     yield scrapy.FormRequest(instance_url, callback=self.parse_api_result,
    #     #                              formdata=form_data, meta={'category': category})
    #     if child_categories != 'No_child':
    #         for child in child_categories.split(";"):
    #             child_url = 'http://fenlei.baike.com/' + urllib.quote(child.encode('utf-8'))
    #             yield scrapy.Request(url=child_url, callback=self.parse)

    # 获取词条的方式每个网站不一样，需要对应修改
    def parse_api_result(self, response):
        json_result = json.loads(response.body)
        instance_list = json_result["list"]
        for instance in instance_list:
            instance_url = instance["title_url"]
            alias = instance["title"]
            yield scrapy.Request(url=instance_url, callback=self.parse_instance,
                                 meta={
                                     'alias': alias,
                                     'category': response.meta['category']
                                 })

    @staticmethod
    def parse_instance(response):
        category = response.meta['category']
        alias = response.meta['alias']

        # 主要修改这下面的内容，将xpath改为Wikipedia网站中的路径
        instance_name = response.xpath('//div[@class="content-h1"]/h1/text()').extract()
        if instance_name:
            instance_name = instance_name[0]

            instance_abstract = response.xpath('//div[@class="summary"]/p').xpath('string(.)').extract()
            if instance_abstract:
                instance_abstract = "".join(instance_abstract)
            else:
                instance_abstract = None

            inner_link = response.xpath('//a[@class="innerlink"]')
            if inner_link:
                instance_internal_link = dict((key, value) for key, value in
                                              zip(inner_link.xpath('text()').extract(),
                                                  inner_link.xpath('@href').extract()))
            else:
                instance_internal_link = None

            infobox = response.xpath('//div[@class="module zoom"]/table/tr/td')
            if infobox:
                instance_infobox = dict((key, value) for key, value in
                                        zip(infobox.xpath('strong/text()').extract(),
                                            infobox.xpath('span').xpath('string(.)').extract()))
            else:
                instance_infobox = None

            content = response.xpath('//div[@id="content"]/p').xpath('string(.)').extract()
            if content:
                instance_content = "".join(content)
            else:
                instance_content = None

            tags = response.xpath('//p[@id="openCatp"]/a')
            if tags:
                instance_tags = dict((key, value) for key, value in
                                     zip(tags.xpath('text()').extract(),
                                         tags.xpath('@href').extract()))
            else:
                instance_tags = None

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
