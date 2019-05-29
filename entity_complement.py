# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/13 20:08
# File usage: 实体信息补全，包含信息框属性抽取，依存句法分析与关联规则


import pandas as pd
import collections
from itertools import chain, combinations

import constant
import data_utils
from nlp_parser import SentenceParser

logger = constant.get_logger("entity_complement")


def get_corpus_in_sentences(doc):
    sentences = [data_utils.clean_text(s) for s in doc.split("。")]
    return sentences


def stats_required_relation(data: pd.DataFrame):
    relation_set = {
        "机构": collections.OrderedDict(),
        "概念": collections.OrderedDict(),
        "人物": collections.OrderedDict(),
        "图书": collections.OrderedDict()
    }

    entity_counter = {
        "机构": 0,
        "概念": 0,
        "人物": 0,
        "图书": 0
    }

    required_relation = {
        "机构": [],
        "概念": [],
        "人物": [],
        "图书": []
    }

    norm_relation = data_utils.load_norm_relation(norm_relation_path)

    for idx, row in data.iterrows():
        entity_counter[row["label"]] += 1
        for relation in [data_utils.clean_text(key) for key in row["instanceInfobox"].keys()]:
            if relation in norm_relation:
                relation = norm_relation[relation]
            if relation in relation_set[row["label"]]:
                relation_set[row["label"]][relation] += 1
            else:
                relation_set[row["label"]][relation] = 1

    with open("data/baike/temp_relation.txt", "w", encoding="utf-8") as f:
        for key in relation_set.keys():
            f.write("=============================\n")
            f.write("%s %d:\n" % (key, entity_counter[key]))
            for item in sorted(relation_set[key].items(), key=lambda d: d[1], reverse=True):
                f.write("%s: %d\n" % (item[0], item[1]))
    f.close()

    for key in relation_set.keys():
        for relation, num in relation_set[key].items():
            if num > entity_counter[key] * 0.3:
                required_relation[key].append(relation)
    return required_relation


def get_item_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        record = record.split(" ")
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))  # Generate 1-itemSets
    return item_set, transaction_list


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


class Apriori(object):
    def __init__(self, min_sup, min_conf):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def return_items_with_min_support(self, item_set, transaction_list, freq_set):

        """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        local_set = collections.defaultdict(int)

        for item in item_set:
            for transaction in transaction_list:
                if item.issubset(transaction):
                    freq_set[item] += 1
                    local_set[item] += 1

        for item, count in local_set.items():
            support = float(count) / len(transaction_list)

            if support >= self.min_sup:
                _itemSet.add(item)

        return _itemSet

    def run_apriori(self, data_iter):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """
        item_set, transaction_list = get_item_list(data_iter)

        freq_set = collections.defaultdict(int)
        large_set = dict()
        # Global dictionary which stores (key=n-itemSets,value=support)
        # which satisfy minSupport

        assocRules = dict()
        # Dictionary which stores Association Rules

        one_c_set = self.return_items_with_min_support(item_set,
                                                       transaction_list,
                                                       freq_set)

        current_l_set = one_c_set
        k = 2
        while current_l_set != set([]):
            large_set[k - 1] = current_l_set
            current_l_set = join_set(current_l_set, k)
            current_c_set = self.return_items_with_min_support(current_l_set,
                                                               transaction_list,
                                                               freq_set)
            current_l_set = current_c_set
            k = k + 1

        def get_support(item):
            """local function which Returns the support of an item"""
            return float(freq_set[item]) / len(transaction_list)

        to_ret_items = []
        for key, value in large_set.items():
            to_ret_items.extend([(tuple(item), get_support(item))
                                 for item in value])

        to_ret_rules = []
        for key, value in large_set.items()[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = get_support(item) / get_support(element)
                        if confidence >= self.min_conf:
                            to_ret_rules.append(((tuple(element), tuple(remain)),
                                                 confidence))
        return to_ret_items, to_ret_rules


def extract_explicit_relation(data: pd.DataFrame, required_relation):
    knowledge_list = []
    deficient_relation = []
    norm_relation = data_utils.load_norm_relation(norm_relation_path)
    for idx, row in data.iterrows():
        required = required_relation[row["label"]]

        knowledge_list.append([row["entityName"], "类别", row["label"]])
        knowledge_list.append([row["entityName"], "简介", data_utils.clean_text(row["instanceAbstract"])])

        if "类别" in required:
            required.remove("类别")
        if "简介" in required:
            required.remove("简介")

        for relation, value in row["instanceInfobox"].items():
            relation = data_utils.clean_text(relation)
            if relation in norm_relation:
                relation = norm_relation[relation]
            value = data_utils.clean_text(value)
            if relation != "" and value != "":
                knowledge_list.append([row["entityName"], relation, value])

            if relation in required:
                required.remove(relation)
        if required:
            deficient_relation.append([row["entityName"], ";".join(required)])

    knowledge_df = pd.DataFrame(knowledge_list, columns=["subject", "predicate", "object"])
    deficient_df = pd.DataFrame(deficient_relation, columns=["entity", "relation"])
    return knowledge_df, deficient_df


def extract_implicit_relation(data: pd.DataFrame):
    candidate_sentences = []
    for idx, row in data.iterrows():
        internal_link = row["instanceInternalLink"].keys()
        sentences = get_corpus_in_sentences(row["instanceContent"])
        for sentence in sentences:
            for candidate_entity in internal_link:
                if candidate_entity in sentence and row["instanceName"] in sentence:
                    candidate_sentences.append([row["instanceName"], candidate_entity, sentence])
    candidate_sentences = pd.DataFrame(candidate_sentences, columns=["entity", "candidate_entity", "sentence"])\
        .drop_duplicates()
    parser = SentenceParser()

    failed_records = []

    with open(implicit_relation_path, "w", encoding="utf-8") as f:
        for idx, row in candidate_sentences.iterrows():
            root = parser.parse_tree(row["sentence"])
            path = root.path([row["entity"], row["candidate_entity"]])
            if path:
                for p in path:
                    result = parser.extract(p)
                    if result:
                        f.write("%s\t%s\n" % (row["sentence"], ";".join(result)))
                    else:
                        failed_records.append([row["entity"], row["candidate_entity"]])
    f.close()

    apriori = Apriori(min_sup=0.5, min_conf=0.8)
    items, strong_correlations = apriori.run_apriori(failed_records)
    with open(strong_correlations_path, "w", encoding="utf-8") as f:
        for correlation in strong_correlations:
            f.write("%s\n" % correlation)
    f.close()


def main():
    original_data = pd.read_json(baike_file_path, orient='records')
    classified_data = pd.read_csv(classified_data_path, sep="\t", names=["entityName", "fakeLabel", "label"])

    data = classified_data.merge(original_data, left_on=["entityName"], right_on=["instanceName"]).dropna()
    required_relation = stats_required_relation(data)

    knowledge, deficient = extract_explicit_relation(data, required_relation)
    knowledge.to_csv(knowledge_data_path, sep="\t", index=None, encoding="utf-8")
    # deficient_data = deficient.merge(original_data, left_on=["entity"], right_on=["instanceName"]).dropna()
    # extract_implicit_relation(deficient_data)


if __name__ == '__main__':
    baike_file_path = 'data/baike/baike_original_data.json'
    classified_data_path = 'data/baike/classified_result.tsv'
    corpus_path = 'data/baike/corpus'
    candidate_entity_vocab_path = 'data/baike/candidate_entity_vocab'
    knowledge_data_path = 'data/KG/knowledge.csv'
    norm_relation_path = "data/baike/norm_relation.csv"
    implicit_relation_path = "data/baike/implicit_relation"
    strong_correlations_path = "data/baike/strong_correlation"
    main()
