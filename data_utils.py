# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2018/12/28 17:13
# File usage: 若干实验的数据处理过程


import collections
import numpy as np
import pandas as pd
import unicodedata
import re

import constant
import jsonlines

MAX_TOKEN_LENGTH = 120
MAX_ATTRIBUTE_LENGTH = 8

logger = constant.get_logger('create_training_data')


def is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_chinese_char(cp):
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)):
        return True

    return False


def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            continue
        else:
            output.append(char)

    output = "".join(output)
    output = re.sub("\（.*?\）|【.*?】", "", output)
    return output


def tokenize_chars(text):
    output = []
    text = clean_text(text)
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(char)
        else:
            continue
    return output


def get_token_vocab(corpus, vocab_path):
    token_list = []
    for doc in corpus:
        token_list.extend(tokenize_chars(doc))
    vocab = create_vocab(token_list, vocab_path)

    logger.info('token vocab length: %s' % len(vocab))


def get_attribute_vocab(infobox: pd.Series, vocab_path):
    attributes_list = []
    infobox = infobox.dropna().tolist()
    for box in infobox:
        attributes_list.extend([clean_text(k) for k in box.keys()])
    vocab = create_vocab(attributes_list, vocab_path)

    logger.info('attribute vocab length: %s' % len(vocab))


def get_category_vocab(category: pd.Series, vocab_path):
    category = category.dropna().tolist()
    vocab = create_vocab(category, vocab_path)

    logger.info('category vocab length: %s' % len(vocab))


def get_entity_vocab(entity: pd.Series, vocab_path):
    entity = entity.dropna().tolist()
    vocab = create_vocab(entity, vocab_path)

    logger.info('entity vocab length: %s' % len(vocab))


def create_vocab(items, vocab_path):
    vocab = collections.OrderedDict()
    idx = 0
    for item in items:
        if item in vocab:
            continue
        else:
            vocab[item] = idx
            idx += 1
    save_to_file(vocab.keys(), vocab_path)
    return vocab


def save_to_file(data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write("%s\n" % d)
    f.close()


def format_corpus(input_path, output_path):
    corpus = []
    with open(input_path, "r", encoding="utf-8") as f:
        doc = f.readline()
        while doc:
            doc = clean_text(doc)
            lines = doc.split("。")
            corpus.append(lines)
            doc = f.readline()
    f.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            for line in doc:
                if line != "":
                    f.write("%s\n" % line)
            f.write("\n")
    f.close()


def load_norm_relation(file_path):
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    norm_relation = {}
    for idx, row in df.iterrows():
        norm_relation[row["relation"]] = row["norm_relation"]

    return norm_relation


def create_entity_classification_data(sample_path, train_data_path, dev_data_path, test_data_path,):
    samples = pd.read_csv(sample_path, sep='\t', encoding='utf-8', dtype=str).dropna()

    def clean(row):
        row["abstract"] = clean_text(row["abstract"])
        row["attribute"] = ";".join([clean_text(attribute) for attribute in row["attribute"].split(";")])
        return row

    samples = samples.apply(clean, axis=1)
    train_idx_end = int(samples.shape[0] * 0.7)
    dev_idx_end = int(samples.shape[0] * 0.2) + train_idx_end

    samples = samples.sample(frac=1)

    samples.iloc[:train_idx_end].to_csv(train_data_path, sep='\t', encoding='utf-8', index=None)
    samples.iloc[train_idx_end:dev_idx_end].to_csv(dev_data_path, sep='\t', encoding='utf-8', index=None)
    # samples.iloc[dev_idx_end:].to_csv(test_data_path, sep='\t', encoding='utf-8', index=None)
    samples.to_csv(test_data_path, sep='\t', encoding='utf-8', index=None)


def get_predict_sample(data: pd.DataFrame, out_path):
    def process(row):
        row["abstract"] = clean_text(row["instanceAbstract"])
        row["attribute"] = ";".join([clean_text(attribute) for attribute in row["instanceInfobox"].keys()])
        row["label"] = "概念"
        return row

    data = data.dropna().apply(process, axis=1).dropna()
    data.to_csv(out_path, columns=["category", "instanceName", "abstract", "attribute", "label"],
                sep='\t', index=None, encoding='utf-8')


def prepare_entity_classification_data():
    df = pd.read_json(baike_file_path, orient='records')
    get_token_vocab(df["instanceAbstract"].dropna().tolist(), token_vocab_path)
    get_attribute_vocab(df["instanceInfobox"], attribute_vocab_path)
    get_category_vocab(df["category"], category_vocab_path)

    create_entity_classification_data(labeled_training_sample_path, train_path, dev_path, test_path)
    get_predict_sample(df[["instanceName", "instanceAbstract", "instanceInfobox", "category"]], predict_sample_path)


def prepare_pre_training_data():
    df = pd.read_csv(knowledge_path, sep="\t", encoding="utf-8", dtype=str)
    get_entity_vocab(df["subject"], entity_vocab_path)
    format_corpus(corpus_path, formatted_corpus_path)


def load_vector_map(file_path):
    entity_vector = {}
    width = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            words = ""
            vector = []
            for feature in item["features"]:
                if feature["token"] == "[CLS]" or feature["token"] == "[SEP]":
                    continue
                else:
                    words += feature["token"]
                    vector.append(feature["layers"][0]["values"])
                    width = len(feature["layers"][0]["values"])
            entity_vector[words] = np.mean(vector, axis=0)

    f.close()
    return entity_vector, width


if __name__ == '__main__':
    baike_file_path = 'data/baike/baike_original_data.json'
    token_vocab_path = 'data/baike/token_vocab'
    attribute_vocab_path = 'data/baike/attribute_vocab'
    category_vocab_path = 'data/baike/category_vocab'
    predict_sample_path = 'data/baike/predict_sample'
    labeled_training_sample_path = 'data/baike/labeled_training_sample.txt'
    train_path = 'data/baike/train_data'
    dev_path = 'data/baike/dev_data'
    test_path = 'data/baike/test_data'

    knowledge_path = 'data/KG/knowledge.csv'
    entity_vocab_path = 'data/KG/entity_vocab'
    corpus_path = 'data/baike/corpus'
    formatted_corpus_path = 'data/bert/corpus'

    pattern_path = 'data/query/sentence_pattern.txt'
    query_path = 'data/query/query.txt'

    # prepare_entity_classification_data()
    # prepare_pre_training_data()
