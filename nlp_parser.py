# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2018/12/7 14:39
# File usage: 依存句法分析方法

from pyltp import SentenceSplitter, Segmentor, Postagger, Parser

from db_connect import DBProcess
from constant import WordNode


CWS_MODEL = "model/cws.model"
POS_MODEL = "model/pos.model"
PARSER_MODEL = "model/parser.model"

db = DBProcess("stock")


class IterDocument(object):
    """
    A class for reading large file memory-friendly
    """

    def __init__(self, path, sep=None):
        """
        :param path: path to the file
        :param sep: delimiter string between fields
        """
        self.path = path
        self.sep = sep

    def __iter__(self):
        """
        :return: iteration in lines
        """
        for line in open(self.path, 'r', encoding="utf-8").readlines():
            # line = line.strip()
            if line == '':
                continue
            if self.sep is not None:
                yield [item for item in line.split(self.sep) if item != ' ' and item != '']
            else:
                yield line


def find_x(l, x):
    return [idx for idx, item in enumerate(l) if item[0] == x]


class SentenceParser:
    """
    A class for sentence analysis
    """

    def __init__(self):
        """
        Load remote lexicon and ltp model
        """
        self.temp_lexicon = "temp_lexicon"
        self.fetch_lexicon()

        self.sentence_splitter = SentenceSplitter()
        self.segment = Segmentor()
        self.segment.load_with_lexicon(CWS_MODEL, self.temp_lexicon)
        self.pos = Postagger()
        self.pos.load_with_lexicon(POS_MODEL, self.temp_lexicon)
        self.tree_parser = Parser()
        self.tree_parser.load(PARSER_MODEL)

        self.rules = IterDocument("data/rule")

    def fetch_lexicon(self):
        """
        Load lexicon and write to local
        """
        res = db.fetch_lexicon()
        with open(self.temp_lexicon, "w", encoding="utf8") as f:
            for item in res:
                token, synonym, norm_token, pos = item
                pos = pos.replace(",", " ")
                token = "%s %s" % (token, pos)
                norm_token = "%s %s" % (norm_token, pos)
                if synonym:
                    synonym = "\n".join(list(map(lambda x: "%s %s" % (x, pos), synonym.split(","))))
                    f.write("%s\n%s\n%s\n" % (token, synonym, norm_token))
                else:
                    f.write("%s\n%s\n" % (token, norm_token))

    def seg_sentence(self, text):
        """
        Segment sentence by punctuation
        :param text: raw string
        :return: vector of sentences, use list() to covert as [sentence0, sentence1, ...]
        """
        return self.sentence_splitter.split(text)

    def seg_token(self, text):
        """
        Segment token by model and lexicon
        :param text: raw string
        :return: vector of tokens use list() to convert as [token0, token1, ...]
        """
        return self.segment.segment(text)

    def pos_tag(self, text):
        """
        Tag position of speech for text by model and lexicon
        :param text: raw string
        :return: vector of pos, use list() to convert as [pos0, pos1, ...]
        """
        tokens = self.seg_token(text)
        return self.pos.postag(tokens)

    def parse_list(self, text, need_info=False):
        """
        Parse the sentence as a list of word node
        :param need_info: whether need extra info
        :param text: raw string
        :return: a list of word node
        """
        result = []
        words = self.seg_token(text)
        pos_list = self.pos.postag(words)
        if len(words) == 0 or len(pos_list) == 0:
            return result
        arcs = self.tree_parser.parse(words, pos_list)

        nodes = list(map(lambda x: (x.head, x.relation), arcs))
        for token, pos, relation in zip(words, pos_list, nodes):
            if need_info:
                info = db.get_word(token)
                if info:
                    category, norm_token, extra = info
                    word_node = WordNode(token, pos, relation[1], category, norm_token, extra)
                else:
                    word_node = WordNode(token, pos, relation[1])
            else:
                word_node = WordNode(token, pos, relation[1])
            result.append(word_node)
        return result

    def parse_tree(self, text, need_info=False):
        """
        Parse the sentence as a dependence tree of word node
        :param need_info: whether need extra info
        :param text: raw string
        :return: a dependence tree of word node
        """
        words = self.seg_token(text)
        pos = self.pos.postag(words)
        if len(words) == 0 or len(pos) == 0:
            return WordNode("", "", "", None)
        arcs = self.tree_parser.parse(words, pos)
        nodes = list(map(lambda x: (x.head, x.relation), arcs))

        root_idx = find_x(nodes, 0)
        if need_info:
            info = db.get_word(words[root_idx[0]])
            if info:
                category, norm_token, extra = info
                root = WordNode(words[root_idx[0]], pos[root_idx[0]], nodes[root_idx[0]][1], category, norm_token,
                                extra)
            else:
                root = WordNode(words[root_idx[0]], pos[root_idx[0]], nodes[root_idx[0]][1])
        else:
            root = WordNode(words[root_idx[0]], pos[root_idx[0]], nodes[root_idx[0]][1])
        tree = {root_idx[0]: root}
        queue = root_idx

        while len(queue):
            next_idx = queue.pop()
            for idx in find_x(nodes, next_idx + 1):
                queue.insert(0, idx)
                if need_info:
                    info = db.get_word(words[root_idx[0]])
                    if info:
                        category, norm_token, extra = info
                        new_node = WordNode(words[idx], pos[idx], nodes[idx][1], category,
                                            norm_token, extra)
                    else:
                        new_node = WordNode(words[idx], pos[idx], nodes[idx][1])
                else:
                    new_node = WordNode(words[idx], pos[idx], nodes[idx][1])
                tree[next_idx].next.append(new_node)
                tree[idx] = new_node
        return root

    def extract(self, path):
        res = []
        for rule in self.rules:
            window_size = len(rule.split(";"))
            if len(path) == window_size:
                if ";".join(map(lambda x: "%s,%s" % (x.relation, x.pos), path)) == rule:
                    res.append(" ".join(map(lambda x: x.token, path)))
            else:
                for i in range(len(path) - window_size):
                    p_slice = ";".join(map(lambda x: "%s,%s" % (x.relation, x.pos), path[i:i + window_size]))
                    if p_slice == rule:
                        res.append(" ".join(map(lambda x: x.token, path[i:i + window_size])))
                        break
        return res


def parse_file(file_path, sep, res_format):
    sp = SentenceParser()
    sentences = IterDocument(file_path, sep)
    res = []
    if res_format == "list":
        for sentence in sentences:
            res.append(sp.parse_list(sentence))
    elif res_format == "tree":
        for sentence in sentences:
            res.append(sp.parse_tree(sentence))
    return res


if __name__ == "__main__":
    parser = SentenceParser()
    r = parser.parse_tree("招行今日股价上涨5.1元")

    all_path = r.path()
    for i, p in enumerate(all_path):
        print("path %s: %s" % (i, "->".join(map(lambda x: "{%s}" % x.to_str(), p))))

    all_path = r.path(["招行", "上涨"])
    for i, p in enumerate(all_path):
        print("path %s: %s" % (i, "->".join(map(lambda x: "{%s}" % x.to_str(), p))))

    # document = "local file"
    # res = parse_file(document, "\t", "list")
