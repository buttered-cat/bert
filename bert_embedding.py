# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
from math import ceil

import tensorflow as tf

if __package__ == '':
    import modeling
    import tokenization
else:
    from . import modeling
    from . import tokenization


class Bert(object):
    def __init__(
            self,
            bert_config_file,
            vocab_file,
            init_checkpoint,
            requested_layers,
            sess=None,
            do_lower_case=True,
            batch_size=32,
            use_one_hot_embeddings=False
    ):
        # tf.logging.set_verbosity(tf.logging.INFO)

        if not sess:
            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)
        else:
            self.sess = sess

        self.layer_indexes = [int(x) for x in requested_layers.split(",")]

        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case
        )

        self.init_checkpoint = init_checkpoint
        self.batch_size = batch_size
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            # input_ids=input_ids,
            # input_mask=input_mask,
            # token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars,
            init_checkpoint
        )

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = %s, shape = %s%s",
                var.name,
                var.shape,
                init_string
            )

        self.sess.run(tf.global_variables_initializer())

    class InputExample(object):

        def __init__(self, unique_id, text_a, text_b):
            self.unique_id = unique_id
            self.text_a = text_a
            self.text_b = text_b

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids

    def convert_examples_to_features(self, examples, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        # Warning: The returned result has lengths of seq_length + 2 because of [CLS] and [SEP].
        # This is different from Google's original code.
        seq_length += 2

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append("[SEP]")
                input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            features.append(
                self.InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def read_examples(self, sentence_list):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        unique_id = 0
        for line in sentence_list:
            line = tokenization.convert_to_unicode(line)

            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                self.InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
        return examples

    def get_embedded_vectors(self, sentence_list):
        """
        Get pre-trained BERT word vectors for each word(character) of each sentence in sentence_list.

        :param sentence_list: A list of strings.
        :return: A list of length max_seq_length + 2, whose form coincide with the original BERT repo, as is shown at the end of this method.

        Better NOT pass too many sentences of great length difference at one call, the redundant padding caused by which may affect performance.
        The returned result has lengths of max_seq_length + 2 because of [CLS] and [SEP].

        TODO: performance optimization: padding inside each batch.
        """
        output = []

        max_seq_length = 0
        for s in sentence_list:
            if len(s) > max_seq_length:
                max_seq_length = len(s)

        if len(sentence_list) == 0:
            return output
        elif max_seq_length <= 0:
            return output

        examples = self.read_examples(sentence_list)

        # features: [InputFeatures]
        features = self.convert_examples_to_features(
            examples=examples, seq_length=max_seq_length, tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        batches = []
        for i in range(ceil(len(features) / self.batch_size)):
            batches.append(features[i * self.batch_size: (i+1) * self.batch_size])

        for batch_features in batches:
            # results: [{"unique_id": int, "layer-wise_outputs": num_layers * max_seq_len * hidden_size}]
            results = self.bert_model.get_encoder_layers(self.sess, batch_features)

            for result in results:
                unique_id = int(result["unique_id"])
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["unique_id"] = unique_id
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(self.layer_indexes):
                        layer_output = result["layer-wise_outputs"][layer_index]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [
                            round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                        ]
                        all_layers.append(layers)
                    features = collections.OrderedDict()
                    features["token"] = token
                    features["layers"] = all_layers
                    all_features.append(features)
                output_json["features"] = all_features
                output.append(output_json)

        # output: [
        #   {
        #       'features': [
        #           {
        #               'token': 'A',
        #               'layers': [
        #                   {
        #                       'index': -1,
        #                       'values': [...] (len == 768)
        #                   }
        #               ]
        #           }
        #       ],
        #       'unique_id': int
        #   },
        #   ...
        # ]
        # includes [CLS] & [SEP]
        return output
