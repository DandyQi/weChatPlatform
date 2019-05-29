# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/4 16:23
# File usage: 问答系统答案抽取，输入为query中实体的文本表示与query的文本表示，输出为相关的知识三元组


import pandas as pd
import configparser
import requests
import json
from get_similar_entity import EntityHash
from db_connect import GraphDBProcess
import data_utils


class KnowledgeSearch(object):
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read("config.conf")
        serving_url = cf.get("serving_url", "LR_model")
        self.tf_serving_url = serving_url
        self.data = pd.read_csv("data/KG/knowledge.csv", sep="\t", encoding="utf-8")
        self.entity_map = EntityHash("data/KG/entity_vector.jsonl", 16)
        self.graph_dbp = GraphDBProcess()
        self.relation_vector_map, _ = data_utils.load_vector_map("data/KG/relation_vector.jsonl")

    def search_by_entity(self, entity_vector, seq_output):
        max_score = 0
        top_similar_entity = self.entity_map.get_similar_entity(entity_vector, 1)
        relations = self.graph_dbp.get_relations_by_entity(top_similar_entity)
        result = (top_similar_entity, "简介")

        for relation in relations:
            score = self.search_relation(seq_output, self.relation_vector_map[relation])
            if score > max_score:
                result = (top_similar_entity, relation)

        knowledge = self.graph_dbp.get_knowledge_by_entity_relation(result)
        return knowledge

    def search_relation(self, seq, relation):
        dict_curl = {
            "seq": seq.tolist(),
            "rel": relation.tolist(),
            "label_id": [0]
        }
        dict_data = {"inputs": dict_curl, "signature_name": "LR_output"}
        json_response = requests.post(self.tf_serving_url, json=dict_data)
        response = json.loads(json_response.content)

        probabilities = response["outputs"]["score"][0][1]

        return probabilities


class AnswerConstruction(object):
    def __init__(self):
        pass

    @staticmethod
    def construct_answer(s, p, o):
        return "%s%s为%s" % (s, p, o)


if __name__ == "__main__":
    pass
