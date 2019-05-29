# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/17 18:09
# File usage: 关系判别模型，输入为query的文本表示与relation的文本表示，输出为相应打分

import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import os

import constant

logger = constant.get_logger("relation_classification")

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("relation_file", "data/query/query_relation.csv", "The path of relation train data")
flags.DEFINE_string("output_dir", "model/relation_classification", "The path for saving model")
flags.DEFINE_integer("valid_size", 400, "The size of valid size")
flags.DEFINE_integer("max_seq_length", 128, "The size of valid size")
flags.DEFINE_integer("width", 768, "The size of valid size")
flags.DEFINE_float("dropout_prob", 0.4, "The probability of dropout")
flags.DEFINE_integer("save_checkpoint_steps", 100, "The num of steps to save checkpoint")
flags.DEFINE_integer("save_summary_steps", 10, "The num of steps to save summary")
flags.DEFINE_integer("batch_size", 32, "The size of each batch")
flags.DEFINE_integer("valid_batch_size", 10, "The size of valid batch")
flags.DEFINE_integer("epoch", 3, "The num of epochs")

flags.DEFINE_bool("do_train", True, "whether to train")
flags.DEFINE_bool("do_valid", True, "whether to valid")


class InputExample(object):
    def __init__(self, guid, seq, rel, label):
        self.guid = guid
        self.seq = seq
        self.rel = rel
        self.label = label


class DataProcessor(object):
    def __init__(self):
        data = pd.read_csv(FLAGS.relation_file, sep="\t", encoding="utf-8", dtype=str)
        data = data.sample(frac=1, replace=True)
        self.train_data = data[FLAGS.valid_size:]
        self.valid_data = data[:FLAGS.valid_size]

    def get_train_example(self):
        logger.info("The size of train data: %s" % self.train_data.shape[0])
        return self.create_example(self.train_data, "train")

    def get_valid_example(self):
        logger.info("The size of valid data: %s" % self.valid_data.shape[0])
        return self.create_example(self.valid_data, "valid")

    @staticmethod
    def create_example(data, set_type):
        examples = []
        for idx, row in data.iterrows():
            guid = "%s-%s" % (set_type, idx)
            seq = np.reshape(row["seq"].split(" "), [FLAGS.max_seq_length, FLAGS.width])
            rel = np.reshape(row["rel"].split(" "), [-1, FLAGS.width])
            label = row["label"]

            examples.append(InputExample(guid, seq, rel, label))

        return examples


def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def convert_example(examples: [InputExample], output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for example in examples:
        features = collections.OrderedDict()
        features["seq"] = create_float_feature(example.seq)
        features["rel"] = create_float_feature(example.rel)
        features["label_id"] = create_int_feature([example.label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()


def input_fn_builder(input_file, is_training):
    logger.info("Load data from %s" % input_file)

    name_to_features = {
        "seq": tf.FixedLenFeature([FLAGS.max_seq_length, FLAGS.width], tf.float32),
        "rel": tf.FixedLenFeature([None, FLAGS.width], tf.float32),
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


def create_model(is_training, seq, rel, label_id):
    dropout_prob = FLAGS.dropout_prob
    if not is_training:
        dropout_prob = 0.0

    with tf.variable_scope("relation_classification"):
        with tf.variable_scope("LR"):
            weights = tf.get_variable(
                name="weights",
                shape=[2, FLAGS.width],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            bias = tf.get_variable(
                name="bias",
                shape=[2],
                initializer=tf.zeros_initializer()
            )

            input_tensor = tf.concat([seq, rel])
            if is_training:
                input_tensor = tf.nn.dropout(input_tensor, keep_prob=(1 - dropout_prob))

            logits = tf.matmul(input_tensor, weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, bias)

            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(label_id, depth=2, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return loss, per_example_loss, logits, probabilities


def model_fn_builder():
    def model_fn(features, labels, mode, params):
        logger.info("***Feature***")
        for feature_name in features.keys():
            logger.info("Feature: %s, Shape: %s\n" % (feature_name, features[feature_name].shape))

        seq = features["seq"]
        rel = features["rel"]
        label_id = features["label_id"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits, probabilities = create_model(
            is_training, seq, rel, label_id
        )

        predict_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "class_ids": predict_classes[:, tf.newaxis],
                "probabilities": probabilities,
                "logits": logits
            }
            export_outputs = {
                "LR_output": tf.estimator.export.PredictOutput(
                    {
                        "score": probabilities
                    }
                )
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

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


def serving_input_receiver_fn():
    label_id = tf.placeholder(tf.int32, [None], name='label_id')
    seq = tf.placeholder(tf.float32, [FLAGS.max_seq_length, FLAGS.width], name='seq')
    rel = tf.placeholder(tf.float32, [None, FLAGS.width], name='rel')

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_id': label_id,
        'seq': seq,
        'rel': rel
    })()
    return input_fn


def main(_):
    processor = DataProcessor()

    model_fn = model_fn_builder()
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
        save_summary_steps=FLAGS.save_summary_steps
    )

    classifier = tf.estimator.Estimator(model_fn=model_fn, params={"batch_size": FLAGS.batch_size}, config=run_config)

    if FLAGS.do_train:
        train_examples = processor.get_train_example()
        train_steps_num = int(len(train_examples) / FLAGS.batch_size * FLAGS.epoch)

        train_file_path = os.path.join(FLAGS.output_dir, "train.tf_record")
        convert_example(train_examples, train_file_path)

        train_input_fn = input_fn_builder(train_file_path, True)
        classifier.train(input_fn=train_input_fn, max_steps=train_steps_num)

    if FLAGS.do_valid:
        valid_examples = processor.get_valid_example()
        valid_steps_num = int(len(valid_examples) / FLAGS.valid_batch_size)

        valid_file_path = os.path.join(FLAGS.output_dir, "valid.tf_record")
        convert_example(valid_examples, valid_file_path)

        valid_input_fn = input_fn_builder(valid_file_path, False)
        result = classifier.evaluate(input_fn=valid_input_fn, steps=valid_steps_num)
        for key in sorted(result.keys()):
            logger.info("   %s: %s" % (key, str(result[key])))

    classifier.export_saved_model(os.path.join(FLAGS.output_dir, "saved_model/1"), serving_input_receiver_fn)


if __name__ == "__main__":
    tf.app.run()
