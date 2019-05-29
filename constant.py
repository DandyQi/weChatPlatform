# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/10 16:48
# File usage: 若干常用类

import logging


def get_logger(name='default'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    custom_format = logging.Formatter(
        '%(name)s: %(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler('%s_log' % name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(custom_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(custom_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Entity(object):
    def __init__(self, entity_id, entity_name, category):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.category = category


class Relation(object):
    def __init__(self, relation_id, relation_name):
        self.relation_id = relation_id
        self.relation_name = relation_name


class Knowledge(object):
    def __init__(self, subject, predicate, obj):
        self.subject = subject
        self.predicate = predicate
        self.object = obj


class GraphInstance(object):
    def __init__(self, label, key, value):
        self.label = label
        self.key = key
        self.value = value

    def __str__(self):
        instance_property = ", ".join(list(map(lambda x: "%s: %s" % (x[0], x[1]), zip(self.key, self.value))))
        return "Label: %s, Property: {%s}" % (self.label, instance_property)


class ParserResponse(object):
    def __init__(self, status, msg, data):
        self.status = status
        self.msg = msg
        self.data = data

    def log(self):
        pass


class WordNode(object):
    """
    A class contains information of words
    """

    def __init__(self, token, pos, relation, category="", norm_token="", extra="", next_nodes=None):
        """
        Initial a word node
        :param token: the word
        :param pos: the word's position of speech
        :param relation: the word's relation in the sentence
        :param category: the category of the word
        :param norm_token: the norm format of the word
        :param extra: the extra info of the word
        :param next_nodes: the next word in the relation
        """
        if next_nodes is None:
            next_nodes = []
        self.token = token
        self.pos = pos
        self.relation = relation
        self.category = category
        self.norm_token = norm_token
        self.extra = extra
        self.next = next_nodes

    def to_str(self):
        """
        :return: the string format of the word node
        """
        return "token: %s, pos: %s, relation: %s, category: %s, norm_token: %s, extra: %s" \
               % (self.token, self.pos, self.relation, self.category, self.norm_token, self.extra)

    def path(self, key_entities=None):
        """
        :return: the path from this word node in the tree
        """

        candidate = []
        queue = self.next.copy()
        path = [self]

        while len(queue):
            cur_node = queue.pop()

            if len(cur_node.next) == 0:
                path.append(cur_node)
                new_path = path.copy()
                candidate.append(new_path)
                path.pop()
            else:
                for node in cur_node.next:
                    queue.insert(0, node)
                path.append(cur_node)

        if key_entities:
            res = []
            for p in candidate:
                flag = False
                entities = [w.token for w in p]
                if isinstance(key_entities, list):
                    for key in key_entities:
                        if key in entities:
                            flag = True
                        else:
                            flag = False
                            break
                else:
                    if key_entities in entities:
                        flag = True
                if flag:
                    res.append(p)
            return res
        else:
            return candidate
