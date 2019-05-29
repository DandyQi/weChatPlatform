# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/4 16:00
# File usage: 问答系统query解析，输入为query，输出为query对应的label，query中包含的entity与relation，entity与query的文本表示向量


import numpy as np
import configparser
import json
import requests
import bert_joint_model
import tokenization

TAG_LIST = ["B-E", "B-R", "I-E", "I-R", "O", "[CLS]", "[SEP]"]
LABEL_LIST = ["Interpret", "StockQuery"]
MAX_SEQ_LENGTH = 128
tokenizer = tokenization.FullTokenizer(vocab_file="model/chinese_L-12_H-768_A-12/vocab.txt")


class QueryParser(object):
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read("config.conf")
        serving_url = cf.get("serving_url", "joint_model")
        self.tf_serving_url = serving_url

        self.tag_map = {}
        for idx, tag in enumerate(TAG_LIST):
            self.tag_map[idx] = tag

        self.label_map = {}
        for idx, label in enumerate(LABEL_LIST):
            self.label_map[idx] = label

    def intent_parse(self, text):
        example = bert_joint_model.InputExample(guid=0, text=text, tag=["O"] * MAX_SEQ_LENGTH, label="Interpret")

        feature = bert_joint_model.convert_single_example(0, example, TAG_LIST, LABEL_LIST, MAX_SEQ_LENGTH, tokenizer)
        input_ids = np.reshape([feature.input_ids], (1, MAX_SEQ_LENGTH))
        input_mask = np.reshape([feature.input_mask], (1, MAX_SEQ_LENGTH))
        segment_ids = np.reshape([feature.segment_ids], (1, MAX_SEQ_LENGTH))
        tag_ids = np.reshape([feature.tag_ids], (1, MAX_SEQ_LENGTH))
        label_ids = [feature.label_id]
        dict_curl = {
            "input_ids": input_ids.tolist(),
            "input_mask": input_mask.tolist(),
            "segment_ids": segment_ids.tolist(),
            "tag_ids": tag_ids.tolist(),
            "label_ids": label_ids
        }
        dict_data = {"inputs": dict_curl, "signature_name": "joint_output"}

        json_response = requests.post(self.tf_serving_url, json=dict_data)
        response = json.loads(json_response.content)

        label = self.label_map[response["outputs"]["label"][0]]

        post_process_text = tokenizer.tokenize(text)

        post_process_text.insert(0, "[CLS]")
        post_process_text.append("[SEP]")

        tag = [self.tag_map[t] for t in response["outputs"]["tag"][0]]
        tag = tag[:tag.index("[SEP]") + 1]

        seq_output = response["output"]["seq_output"][0][:tag.index("[SEP]") + 1]

        entity = ""
        relation = ""
        entity_vector = []
        for word, tag, vector in zip(post_process_text, tag, seq_output):
            if tag == "B-E" or tag == "I-E":
                entity += word
                entity_vector.append(vector)
            if tag == "B-R" or tag == "I-R":
                relation += word
        if entity != "" and relation == "":
            relation = "简介"

        entity_vector = np.mean(entity_vector, axis=0)

        return label, entity, relation, entity_vector, seq_output


if __name__ == "__main__":
    query = "上升三角形代表什么含义"
    qp = QueryParser()
    i, e, r, v, o = qp.intent_parse(query)
    print(i, e, r, v.shape, o.shape)
