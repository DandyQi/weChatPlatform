# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/6 15:24
# File usage: 实体分类模型，输入为词条的摘要、信息框属性与原始类别，输出为实体对应类别

import tensorflow as tf
import pandas as pd
import collections
import os

import data_utils
import constant

logger = constant.get_logger("entity_classification")
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", "data/baike/train_data", "Input train file")
flags.DEFINE_string("dev_file", "data/baike/dev_data", "Input dev file")
flags.DEFINE_string("test_file", "data/baike/test_data", "Input test file")
flags.DEFINE_string("output_dir", "model/entity_classification", "Output directory for model checkpoints")
flags.DEFINE_string("token_vocab", "data/baike/token_vocab", "Path to token vocab")
flags.DEFINE_string("attribute_vocab", "data/baike/attribute_vocab", "Path to attribute vocab")
flags.DEFINE_string("category_vocab", "data/baike/category_vocab", "Path to category vocab")

flags.DEFINE_bool("do_train", True, "Whether to train")
flags.DEFINE_bool("do_eval", True, "Whether to eval")
flags.DEFINE_bool("do_predict", True, "Whether to predict")

flags.DEFINE_integer("epoch", 5, "train epoch")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_integer("eval_batch_size", 8, "Batch size for evaluation")
flags.DEFINE_integer("save_steps", 5, "Number of steps to save checkpoints")
flags.DEFINE_integer("max_token_length", 120, "The max length of tokens")
flags.DEFINE_integer("max_attribute_length", 8, "The max length of attributes")
flags.DEFINE_float("dropout_prob", 0.4, "Dropout probability")
flags.DEFINE_integer("token_embedding_size", 50, "Size of the token embedding")
flags.DEFINE_integer("attribute_embedding_size", 20, "Size of the attribute embedding")
flags.DEFINE_list("filter_sizes", [2, 3, 4], "Size of convolution filters")
flags.DEFINE_integer("filter_num", 100, "Number of convolution filters")
flags.DEFINE_integer("projection_units", 50, "Number of projection units")
flags.DEFINE_list("hidden_units", [512, 256, 128], "Size of the hidden units")


class InputExample(object):
    def __init__(self, example_name, tokens, attributes, category, label):
        self.example_name = example_name
        self.tokens = tokens
        self.attributes = attributes
        self.category = category
        self.label = label


class DataProcessor(object):
    def get_train_examples(self):
        data = pd.read_csv(FLAGS.train_file, sep="\t", encoding="utf-8").dropna()
        logger.info("The size of train data: %d" % data.shape[0])
        return self.create_example(data)
    
    def get_dev_examples(self):
        data = pd.read_csv(FLAGS.dev_file, sep="\t", encoding="utf-8").dropna()
        logger.info("The size of dev data: %d" % data.shape[0])
        return self.create_example(data)
    
    def get_test_examples(self):
        data = pd.read_csv(FLAGS.test_file, sep="\t", encoding="utf-8").dropna()
        logger.info("The size of test data: %d" % data.shape[0])
        return self.create_example(data)

    @staticmethod
    def create_example(data):
        examples = []
        for idx, row in data.iterrows():
            examples.append(InputExample(
                example_name=row["instanceName"],
                tokens=data_utils.tokenize_chars(row["abstract"]),
                attributes=row["attribute"].split(";"),
                category=row["category"],
                label=row["label"]
            ))
        return examples
    
    @staticmethod
    def get_labels():
        return ["人物", "机构", "概念", "图书"]

    def get_token_vocab(self):
        token_vocab = self.load_vocab(FLAGS.token_vocab)
        logger.info("The size of token vocab: %d" % len(token_vocab))
        return token_vocab, len(token_vocab)

    def get_attribute_vocab(self):
        attribute_vocab = self.load_vocab(FLAGS.attribute_vocab)
        logger.info("The size of attribute vocab: %d" % len(attribute_vocab))
        return attribute_vocab, len(attribute_vocab)

    def get_category_vocab(self):
        category_vocab = self.load_vocab(FLAGS.category_vocab)
        logger.info("The size of category vocab: %d" % len(category_vocab))
        return category_vocab, len(category_vocab)

    @staticmethod
    def load_vocab(vocab_path):
        vocab = collections.OrderedDict()
        vocab[" "] = 0
        idx = 1
        with open(vocab_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.strip()
                vocab[line] = idx
                idx += 1
                line = f.readline()
        return vocab
    
    
def truncate_length(input_list, max_length):
    if len(input_list) > max_length:
        output = input_list[:max_length]
    else:
        output = input_list
        while len(output) < max_length:
            output.append(0)
    return output


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def convert_examples_to_features(
        examples,
        token_map,
        attribute_map,
        category_map,
        label_map,
        max_token_length,
        max_attribute_length,
        output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    def convert_single_example(i, item):
        token_ids = [token_map[token] for token in item.tokens]
        attribute_ids = [attribute_map[attribute] for attribute in item.attributes]
        category_id = category_map[item.category]
        label_id = label_map[item.label]
        
        token_ids = truncate_length(token_ids, max_token_length)
        attribute_ids = truncate_length(attribute_ids, max_attribute_length)

        features = collections.OrderedDict()
        features["token_ids"] = create_int_feature(token_ids)
        features["attribute_ids"] = create_int_feature(attribute_ids)
        features["category_id"] = create_int_feature([category_id])
        features["label_id"] = create_int_feature([label_id])
        
        if i < 3:
            logger.info("***Example***")
            for feature_name in features.keys():
                f = features[feature_name]
                values = f.int64_list.value
                logger.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))
                
        return tf.train.Example(features=tf.train.Features(feature=features))

    for idx, example in enumerate(examples):
        tf_example = convert_single_example(idx, example)
        writer.write(tf_example.SerializeToString())
    writer.close()


def input_fn_builder(input_file, max_token_length, max_attribute_length, is_training):

    logger.info("Load data from %s" % input_file)

    name_to_features = {
        "token_ids": tf.FixedLenFeature([max_token_length], tf.int64),
        "attribute_ids": tf.FixedLenFeature([max_attribute_length], tf.int64),
        "category_id": tf.FixedLenFeature([], tf.int64),
        "label_id": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, features):
        example = tf.parse_single_example(record, features)

        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        drop_remainder = False
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            drop_remainder = True

        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn
    

def model_fn_builder(label_num, token_vocab_size, attribute_vocab_size, category_vocab_size):
    def model_fn(features, labels, mode, params):
        logger.info("***Features***")
        for feature_name in features.keys():
            logger.info("Feature: %s, Shape: %s\n" % (feature_name, features[feature_name].shape))

        token_ids = features["token_ids"]
        attribute_ids = features["attribute_ids"]
        category_id = features["category_id"]
        label_id = features["label_id"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        total_loss, per_example_loss, logits, probabilities = create_model(
            is_training, token_ids, attribute_ids, category_id, label_id,
            label_num, token_vocab_size, attribute_vocab_size, category_vocab_size)

        predict_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "class_ids": predict_classes[:, tf.newaxis],
                "probabilities": probabilities,
                "logits": logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        precision = tf.metrics.precision(label_id, predict_classes, name="precision")
        recall = tf.metrics.recall(label_id, predict_classes, name="recall")

        tf.summary.scalar("precision", precision[1])
        tf.summary.scalar("recall", recall[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                "precision": precision,
                "recall": recall
            }
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

    return model_fn

        
def create_model(is_training, token_ids, attribute_ids, category_id, label_id,
                 label_num, token_vocab_size, attribute_vocab_size, category_vocab_size):
    dropout_prob = FLAGS.dropout_prob
    if not is_training:
        dropout_prob = 0.0
        
    with tf.variable_scope("entity_classification"):
        with tf.variable_scope("embeddings"):
            token_embedding, token_embedding_table = embedding_lookup(
                input_ids=token_ids,
                vocab_size=token_vocab_size,
                embedding_size=FLAGS.token_embedding_size,
                initializer_range=0.02,
                embedding_name="token_embedding"
            )
            attribute_embedding, attribute_embedding_table = embedding_lookup(
                input_ids=attribute_ids,
                vocab_size=attribute_vocab_size,
                embedding_size=FLAGS.attribute_embedding_size,
                initializer_range=0.02,
                embedding_name="attribute_embedding"
            )
            category_embedding, category_embedding_table = embedding_lookup(
                input_ids=category_id,
                vocab_size=category_vocab_size,
                embedding_size=None,
                initializer_range=0.02,
                embedding_name="category_embedding"
            )

        with tf.variable_scope("convolution"):
            token_shape = get_shape_list(token_embedding)

            conv_outputs = []
            for filter_size in FLAGS.filter_sizes:
                conv = tf.layers.conv1d(
                    inputs=token_embedding,
                    filters=FLAGS.filter_num,
                    kernel_size=filter_size
                )
                conv = tf.nn.relu(conv)
                conv = dropout(tf.reduce_max(conv, axis=1), dropout_prob)
                conv = tf.reshape(conv, [token_shape[0], FLAGS.filter_num])
                conv_outputs.append(conv)
            conv_output = tf.concat(conv_outputs, axis=-1)

        with tf.variable_scope("projection"):
            attribute_shape = get_shape_list(attribute_embedding)
            attribute_projection = tf.layers.dense(attribute_embedding, FLAGS.projection_units, tf.nn.relu)
            attribute_projection = dropout(attribute_projection, dropout_prob)
            attribute_projection = tf.reshape(tf.reduce_sum(attribute_projection, axis=1),
                                              [attribute_shape[0], FLAGS.projection_units])

            category_shape = get_shape_list(category_embedding)
            category_projection = tf.layers.dense(category_embedding, FLAGS.projection_units, tf.nn.relu)
            category_projection = dropout(category_projection, dropout_prob)
            category_projection = tf.reshape(category_projection,
                                             [category_shape[0], FLAGS.projection_units])

        with tf.variable_scope("classify"):
            feature_map = tf.concat([conv_output, attribute_projection, category_projection], axis=-1)
            for unit in FLAGS.hidden_units:
                feature_map = tf.layers.dense(feature_map, unit, tf.nn.relu)

            logits = tf.layers.dense(feature_map, label_num, None)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            
            one_hot_labels = tf.one_hot(label_id, depth=label_num, dtype=tf.float32)
            
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            
            return loss, per_example_loss, logits, probabilities
        
    
def embedding_lookup(input_ids, vocab_size, embedding_size, initializer_range, embedding_name):
    if input_ids.shape.ndims == 1:
        embedding = tf.one_hot(input_ids, depth=vocab_size, dtype=tf.float32)
        return embedding, None

    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    embedding = tf.nn.embedding_lookup(embedding_table, input_ids)
    input_shape = get_shape_list(input_ids)
    embedding = tf.reshape(embedding, input_shape[0:-1] + [input_shape[-1] * embedding_size])
    
    return embedding, embedding_table


def create_initializer(initializer_range):
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    else:
        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output
    

def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    
    non_static_indexes = []
    for idx, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(idx)
    
    if not non_static_indexes:
        return shape
    
    dyn_shape = tf.shape(tensor)
    for idx in non_static_indexes:
        shape[idx] = dyn_shape[idx]
        
    return shape


def main(_):
    data_processor = DataProcessor()
    label_list = data_processor.get_labels()

    label_vocab = {}
    label_vocab_ids = {}
    for idx, label in enumerate(label_list):
        label_vocab[label] = idx
        label_vocab_ids[idx] = label

    label_num = len(label_list)
    token_vocab, token_vocab_size = data_processor.get_token_vocab()
    attribute_vocab, attribute_vocab_size = data_processor.get_attribute_vocab()
    category_vocab, category_vocab_size = data_processor.get_category_vocab()

    model_fn = model_fn_builder(label_num, token_vocab_size, attribute_vocab_size, category_vocab_size)
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir,
                                        save_checkpoints_steps=FLAGS.save_steps,
                                        save_summary_steps=FLAGS.save_steps)

    classifier = tf.estimator.Estimator(model_fn=model_fn, params={"batch_size": FLAGS.batch_size}, config=run_config)

    if FLAGS.do_train:
        train_examples = data_processor.get_train_examples()
        train_steps_num = int(len(train_examples) / FLAGS.batch_size * FLAGS.epoch)

        train_file_path = os.path.join(FLAGS.output_dir, "train.tf_record")
        convert_examples_to_features(train_examples, token_vocab, attribute_vocab, category_vocab, label_vocab,
                                     FLAGS.max_token_length, FLAGS.max_attribute_length, train_file_path)

        train_input_fn = input_fn_builder(train_file_path, FLAGS.max_token_length, FLAGS.max_attribute_length, True)
        classifier.train(input_fn=train_input_fn, max_steps=train_steps_num)

    if FLAGS.do_eval:
        eval_examples = data_processor.get_dev_examples()
        eval_steps_num = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_file_path = os.path.join(FLAGS.output_dir, "dev.tf_record")
        convert_examples_to_features(eval_examples, token_vocab, attribute_vocab, category_vocab, label_vocab,
                                     FLAGS.max_token_length, FLAGS.max_attribute_length, eval_file_path)

        eval_input_fn = input_fn_builder(eval_file_path, FLAGS.max_token_length, FLAGS.max_attribute_length, False)
        result = classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps_num)
        for key in sorted(result.keys()):
            logger.info("   %s: %s" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = data_processor.get_test_examples()
        predict_file_path = os.path.join(FLAGS.output_dir, "test.tf_record")
        convert_examples_to_features(predict_examples, token_vocab, attribute_vocab, category_vocab, label_vocab,
                                     FLAGS.max_token_length, FLAGS.max_attribute_length, predict_file_path)
        predict_input_fn = input_fn_builder(predict_file_path, FLAGS.max_token_length,
                                            FLAGS.max_attribute_length, False)
        result = classifier.predict(predict_input_fn)

        predict_output_path = os.path.join(FLAGS.output_dir, "predict_result.tsv")
        with tf.gfile.GFile(predict_output_path, "w") as writer:
            for prediction, example in zip(result, predict_examples):
                predict_class_id = prediction["class_ids"][0]
                predict_class = label_vocab_ids[predict_class_id]

                line = "%s\t%s\t%s\n" % (example.example_name, example.label, predict_class)
                writer.write(line)
            writer.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("train_file")
    flags.mark_flag_as_required("dev_file")
    flags.mark_flag_as_required("test_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
