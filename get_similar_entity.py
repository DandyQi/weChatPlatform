# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/5 10:43
# File usage: 构建实体hash存储，并提供相似实体查询方法

import annoy
import data_utils


class EntityHash(object):
    def __init__(self, entity_features_path, k):
        self.words_map = {}
        self.words_map_inv = {}

        self.entity_vector, self.width = data_utils.load_vector_map(entity_features_path)

        self.hash_map = annoy.AnnoyIndex(self.width)
        self.create_hash(k)

    def create_hash(self, k):
        for idx, word in enumerate(self.entity_vector.keys()):
            self.words_map[word] = idx
            self.words_map_inv[idx] = word

        for word, vector in self.entity_vector.items():
            idx = self.words_map[word]
            self.hash_map.add_item(idx, vector)

        self.hash_map.build(k)

    def get_similar_entity(self, query_vector, n=6):
        if query_vector.shape[0] != self.width:
            raise ValueError("The length of the vector have to equal %s" % str(self.width))

        top_n_entity_idx = self.hash_map.get_nns_by_vector(query_vector, n)
        top_n_entity = [self.words_map_inv[idx] for idx in top_n_entity_idx]

        return top_n_entity


if __name__ == "__main__":
    eh = EntityHash("data/KG/entity_vector.jsonl", 16)
    test_entity = "绩优股"
    print(eh.get_similar_entity(eh.entity_vector[test_entity]))
