# -*- coding: utf-8 -*-

# Author: Dandy Qi
# Created time: 2019/1/29 21:51
# File usage: 多任务联合意图解析模型，输入为query，输出为tag与label


import os
import collections

import pandas as pd
import tensorflow as tf

import constant
import modeling
import optimization
import tokenization

logger = constant.get_logger("joint_model")

flags = tf.flags
FLAGS = flags.FLAGS

# 输入文件格式为包含query，tag与label三列的制表符分隔csv文件
# 其中query为未分词的句子，tag对应query中的每个字符，以空格分隔
# 一行数据形式如：招行的总部在哪？\tB-E I-E O B-R I-R O O\tInterpret

flags.DEFINE_string("query_file", "data/query/query.txt", "Input query file")

flags.DEFINE_string("bert_config_file", "model/chinese_L-12_H-768_A-12/bert_config.json",
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "model/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", "model/joint_model",
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("valid_size", 200, "The size of valid data set")

flags.DEFINE_string(
    "init_checkpoint", "model/chinese_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    def __init__(self, guid, text, tag, label):
        self.guid = guid
        self.text = text
        self.tag = tag
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, tag_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_ids = tag_ids
        self.label_id = label_id


class DataProcessor(object):
    def __init__(self):
        data = pd.read_csv(FLAGS.query_file, sep="\t", encoding="utf-8", dtype=str, names=["text", "tag", "label"])
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
    def get_label():
        return ["Interpret", "StockQuery"]

    @staticmethod
    def get_tag():
        return ["B-E", "B-R", "I-E", "I-R", "O", "[CLS]", "[SEP]"]

    @staticmethod
    def create_example(data, set_type):
        examples = []
        for idx, row in data.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text = tokenization.convert_to_unicode(row["text"])
            tag = row["tag"].split(" ")
            label = row["label"]

            examples.append(InputExample(guid=guid, text=text, tag=tag, label=label))

        return examples


def convert_single_example(ex_index, example: InputExample, tag_list: list, label_list: list, max_seq_length,
                           tokenizer: tokenization.FullTokenizer):
    query = tokenizer.tokenize(example.text)

    if len(query) > max_seq_length - 2:
        query = query[0:(max_seq_length - 2)]

    tokens = ["[CLS]"]
    tags = ["[CLS]"]
    for idx, token in enumerate(query):
        tokens.append(token)
        tags.append(example.tag[idx])
    tokens.append("[SEP]")
    tags.append("[SEP]")
    segment_ids = [0] * len(tokens)

    tag_map = {}
    for idx, tag in enumerate(tag_list):
        tag_map[tag] = idx
    label_map = {}
    for idx, label in enumerate(label_list):
        label_map[label] = idx

    tag_ids = [tag_map[tag] for tag in tags]
    label_id = label_map[example.label]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        tag_ids.append(0)

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % example.guid)
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("tag: %s" % " ".join(tags))
        logger.info("label: %s" % example.label)

    feature = InputFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        tag_ids=tag_ids,
        label_id=label_id
    )

    return feature


def convert_examples_to_features(examples, tag_list, label_list, max_seq_length, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, tag_list, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["tag_ids"] = create_int_feature(feature.tag_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tag_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, keys):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, keys)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def serving_input_receiver_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    tag_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='tag_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'tag_ids': tag_ids
    })()
    return input_fn


def get_output(input_tensor, num_output, outputs, output_type, is_training=True):
    hidden_size = input_tensor.shape[-1].value
    output_weights = tf.get_variable(
        "%s_output_weights" % output_type,
        shape=[num_output, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "%s_output_bias" % output_type,
        [num_output],
        initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("%s_loss" % output_type):
        if is_training:
            input_tensor = tf.nn.dropout(input_tensor, keep_prob=0.9)
        if output_type == "tag":
            input_tensor = tf.reshape(input_tensor, [-1, hidden_size])
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        if output_type == "tag":
            logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_output])

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(outputs, depth=num_output, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_labels, num_tags):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info(" name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        tag_ids = features["tag_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        label_loss, label_per_example_loss, label_logits, label_probs = get_output(
            model.get_pooled_output(), num_labels, label_ids, "label", is_training
        )

        tag_loss, tag_per_example_loss, tag_logtis, tag_probs = get_output(
            model.get_sequence_output(), num_tags, tag_ids, "tag", is_training
        )

        tag_pred = tf.argmax(tag_logtis, axis=2)
        label_pred = tf.argmax(label_logits, axis=1)

        total_loss = label_loss + tag_loss
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint
        )
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logger.info("*** Trainable Variables ***")
        for var in tvars:
            init_strings = ""
            if var.name in initialized_variable_names:
                init_strings = ", *INIT_FROM_CKPT*"
            logger.info(" name = %s, shape = %s%s" % (var.name, var.shape, init_strings))
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=None)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_per_loss, tag_per_loss, label, tag, label_p, tag_p):
                label_precision = tf.metrics.precision(label, label_p,
                                                       name="label_precision")
                label_recall = tf.metrics.recall(label_ids, label_p, name="label_recall")
                tag_precision = tf.metrics.precision(tag, tag_p, name="tag_precision")
                tag_recall = tf.metrics.recall(tag, tag_p, name="tag_recall")

                eval_label_loss = tf.metrics.mean(values=label_per_loss)
                eval_tag_loss = tf.metrics.mean(values=tag_per_loss)

                return {
                    "label_precision": label_precision,
                    "label_recall": label_recall,
                    "tag_precision": tag_precision,
                    "tag_recall": tag_recall,
                    "eval_label_loss": eval_label_loss,
                    "eval_tag_loss": eval_tag_loss
                }
            eval_metrics = (metric_fn, [label_per_example_loss,
                                        tag_per_example_loss,
                                        label_ids,
                                        tag_ids,
                                        label_pred,
                                        tag_pred])

            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=None
            )
        else:
            sequence_output = model.get_sequence_output()
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "tag": tag_pred,
                    "label": label_pred
                },
                export_outputs={
                    "joint_output": tf.estimator.export.PredictOutput(
                        {
                            "tag": tag_pred,
                            "label": label_pred,
                            "seq_output": sequence_output
                        }
                    )
                }
            )
        return output_spec

    return model_fn


def main(_):
    processor = DataProcessor()

    label = processor.get_label()
    num_labels = len(label)
    tag = processor.get_tag()
    num_tags = len(tag)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_example()
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        num_tags=num_tags,
        num_labels=num_labels
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size, )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        convert_examples_to_features(
            examples=train_examples,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer,
            output_file=train_file,
            tag_list=tag,
            label_list=label
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_valid_example()
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        convert_examples_to_features(
            examples=eval_examples,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer,
            output_file=eval_file,
            tag_list=tag,
            label_list=label
        )

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    estimator._export_to_tpu = False
    estimator.export_saved_model(os.path.join(FLAGS.output_dir, "saved_model/1"), serving_input_receiver_fn)


if __name__ == "__main__":
    tf.app.run()
