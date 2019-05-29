# -*- coding:utf-8 -*-

# Author: Dandy Qi
# Created time: 2018/12/7 14:39
# File usage: 问答控制流程，输入为query，输出为知识三元组


from __future__ import print_function

import urllib.request as ur
from urllib.request import quote
import json

from FQAParser import QueryParser
from FQAProvider import KnowledgeSearch, AnswerConstruction


class QAProcess:
    def __init__(self):
        api_key = "025e2012d96d43e89a8e6738ef27e4e3"
        self.turing_url = "http://www.tuling123.com/openapi/api?key=%s&info=" % api_key
        self.parser = QueryParser()
        self.ks = KnowledgeSearch()
        self.ac = AnswerConstruction()

    def response(self, query):
        intent, entity, relation, entity_vector, seq_output = self.parser.intent_parse(query)
        if intent == "Interpret" and entity != "":
            s, p, o = self.ks.search_by_entity(entity_vector, seq_output)
            if s:
                answer = self.ac.construct_answer(s, p, o)

                response = {
                    "query": query,
                    "intent": intent,
                    "extract_entity": entity,
                    "extract_relation": relation,
                    "retrieval_entity": s,
                    "retrieval_relation": p,
                    "answer": answer
                }
                return response
            else:
                answer = self.turing_response(query)

                response = {
                    "query": query,
                    "intent": intent,
                    "extract_entity": entity,
                    "extract_relation": relation,
                    "retrieval_entity": s,
                    "retrieval_relation": p,
                    "answer": answer
                }
                return response
        else:
            answer = self.turing_response(query)
            response = {
                "query": query,
                "intent": "Other",
                "extract_entity": entity,
                "extract_relation": relation,
                "retrieval_entity": "",
                "retrieval_relation": "",
                "answer": answer
            }
            return response

    def turing_response(self, query):
        url = "%s%s" % (self.turing_url, quote(query))
        res = ur.urlopen(url).read()
        res_json = json.loads(res)

        content = res_json["text"]

        return content


if __name__ == "__main__":
    qa = QAProcess()
    a = qa.response("股东的定义是啥")
    print(a)
