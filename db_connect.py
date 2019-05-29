# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2018/12/7 14:42
# File usage: MySQL与Neo4j数据库的访问方式


import pymysql
import configparser
from neo4j import GraphDatabase

from constant import GraphInstance


class DBProcess:
    def __init__(self, domain):
        cf = configparser.ConfigParser()
        cf.read("config.conf")

        db_host = cf.get("db", "db_remote_host")
        db_user = cf.get("db", "db_user")
        db_password = cf.get("db", "db_password")
        db_database = cf.get("db", "db_database")

        self.db = pymysql.connect(db_host, db_user, db_password, db_database)
        self.domain = domain

    def get_word(self, token):
        sql = "SELECT category, norm_token, extra " \
              "FROM entity " \
              "WHERE (token='%s' OR norm_token='%s' OR find_in_set('%s', synonym)) and domain='%s'" \
              "UNION " \
              "SELECT category, norm_token, extra " \
              "FROM relation " \
              "WHERE (token='%s' OR norm_token='%s' OR find_in_set('%s', synonym)) and domain='%s'" \
              % (token, token, token, self.domain, token, token, token, self.domain)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            return result
        except Exception as e:
            print(sql)
            print(e)

    def fetch_lexicon(self):
        sql = "SELECT token, synonym, norm_token, pos FROM entity WHERE domain='%s'" \
              "UNION SELECT token, synonym, norm_token, pos FROM relation WHERE domain='%s'" \
              % (self.domain, self.domain)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(sql)
            print(e)

    def get_extra(self, key):
        sql = "SELECT extra FROM entity " \
              "WHERE (token='%s' OR norm_token='%s' OR find_in_set('%s', synonym)) AND domain='%s'" \
              % (key, key, key, self.domain)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            return result
        except Exception as e:
            print(sql)
            print(e)


class GraphDBProcess:
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read("../config.conf")

        db_uri = cf.get("graph_db", "db_remote_uri")
        db_user = cf.get("graph_db", "db_user")
        db_password = cf.get("graph_db", "db_password")

        self.driver = GraphDatabase.driver(db_uri, auth=(db_user, db_password))

    def close(self):
        self.driver.close()

    def create_node(self, node):
        """

        :rtype: object
        """
        with self.driver.session() as session:
            message = session.write_transaction(self._create_node, node)
            return message

    def create_relationship(self, sub_node, obj_node, relationship):
        with self.driver.session() as session:
            message = session.write_transaction(self._create_relationship, sub_node, obj_node, relationship)
            return message

    def delete_stock_node(self):
        with self.driver.session() as session:
            message = session.write_transaction(self._delete_stock_node)
            return message

    def get_relations_by_entity(self, entity):
        with self.driver.session() as session:
            records = session.write_transaction(self._query_relation, entity)
            if records:
                return [record["r.name"] for record in records]
            else:
                return []

    def get_knowledge_by_entity_relation(self, key):
        (entity, relation) = key
        with self.driver.session() as session:
            records = session.write_transaction(self._query_knowledge, entity, relation)
            if records:
                return [(entity, relation, record["o.name"]) for record in records]
            else:
                return []

    @staticmethod
    def _create_node(tx, node: GraphInstance):
        try:
            statement = "CREATE (node:%s) " % node.label
            for item in zip(node.key, node.value):
                statement += "SET node.%s = '%s' " % (item[0], item[1])
            tx.run(statement)
            return "success"
        except Exception as e:
            return e

    @staticmethod
    def _create_relationship(tx, sub_node, obj_node, relationship: GraphInstance):
        try:
            statement = "MATCH (s:%s), (o:%s) " % (sub_node["label"], obj_node["label"])
            statement += "WHERE s.%s = '%s' and o.%s = '%s' " \
                         % (sub_node["key"], sub_node["value"], obj_node["key"], obj_node["value"])
            if len(relationship.key) != 0:
                statement += "CREATE (s)-[r:%s {" % relationship.label
                statement += ", ".join(list(map(lambda key, value: "%s: '%s'" % (key, value),
                                                relationship.key, relationship.value)))
                statement += "}]->(o) "
            else:
                statement += "CREATE (s)-[r:%s]->(o) " % relationship.label
            tx.run(statement)
            return "success"
        except Exception as e:
            return e

    @staticmethod
    def _delete_stock_node(tx):
        try:
            statement = "MATCH (s:StockCompany) DETACH DELETE s"
            tx.run(statement)
            return "success"
        except Exception as e:
            return e

    @staticmethod
    def _query_relation(tx, sub_node):
        statement = "MATCH (s)-[r]->(o) WHERE s.name = $name RETURN r.name"
        records = tx.run(statement, name=sub_node)
        return records

    @staticmethod
    def _query_knowledge(tx, sub_node, relation):
        statement = "MATCH (s)-[r]->(o) WHERE s.name = $s_name AND r.name = $r_name RETURN o.content"
        records = tx.run(statement, s_name=sub_node, r_name=relation)
        return records


if __name__ == "__main__":
    # db = DBProcess()
    # res = db.fetch_lexicon("stock")
    #
    # print(res)
    graph_db = GraphDBProcess()
    graph_db.create_relationship({"label": "Province", "key": "name", "value": "北京"},
                                 {"label": "Area", "key": "name", "value": "华北"},
                                 {"label": "Located", "key": ["name", "test"], "value": ["位于", "t"]})
