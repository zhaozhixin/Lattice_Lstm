import tensorflow as tf
import numpy as np
from Configure import *
from utils.data import Data
from tfnlp.cell.lattice_lstm import LatticeLSTMCell
import sys
from tensorflow.python import debug as tf_debug

import random


class LatticeNet:
    def __init__(self, conf=None, model_save_path=None):
        lexicon_num_units = 64
        char_num_units = 64

        self.configure = conf

        self.batch_size = None
        self.max_char_len = None
        self.max_lexicon_words_num = None
        self.num_units = None
        self.num_tags = None

        self.gaz_file = None
        self.char_emb = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None

        def my_filter_callable(datum, tensor):
            # A filter that detects zero-valued scalars.
            return len(tensor.shape) == 0 and tensor == 0.0

        self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
        self.sess.add_tensor_filter('my_filter', my_filter_callable)

        self.sess = tf.Session()
        self.placeholders = {}
        self.epoch = 0
        self.loss = None
        self.train_op = None

        self.set_configure() if self.configure else None

        self.bichar_emb = None

    def set_configure(self):
        for key in self.configure.confs:
            self.__dict__[key] = self.configure[key]
            #print(key,self.__dict__[key])


    def create_model(self):
        print(self.data.pretrain_word_embedding)
        char_embeddings = tf.Variable(self.data.pretrain_word_embedding, dtype=tf.float32, name="char_embeddings")
        word_embeddings = tf.Variable(self.data.pretrain_gaz_embedding, dtype=tf.float32, name="word_embeddings")
        #char_embeddings = tf.get_variable("ce", [1000, 50] ,dtype=tf.float32)
        #word_embeddings = tf.get_variable("we", [1500, 50] ,dtype=tf.float32)


        char_ids = tf.placeholder(tf.int32, [None, self.max_char_len])
        lexicon_word_ids = tf.placeholder(tf.int32, [None, self.max_char_len, self.max_lexicon_words_num])
        word_length_tensor = tf.placeholder(tf.float32, [None, self.max_char_len,self.max_lexicon_words_num])

        labels = tf.placeholder(tf.int32, [None, self.max_char_len])

        #labels_one_hot = tf.one_hot(labels, self.num_tags, 1, 0, axis=-1)
        #char_seq_len = tf.placeholder(tf.int32, [None])


        lexicon_word_ids_reshape = tf.reshape(lexicon_word_ids, [-1, self.max_char_len * self.max_lexicon_words_num])
        lexicon_word_embed_reshape = tf.nn.embedding_lookup(word_embeddings, lexicon_word_ids_reshape)
        lexicon_word_embed = tf.reshape(lexicon_word_embed_reshape, [-1, self.max_char_len , self.max_lexicon_words_num, self.emb_size])

        char_embed = tf.nn.embedding_lookup(char_embeddings, char_ids)

        ner_lattice_lstm = LatticeLSTMCell(self.num_units,
                                           self.num_units,
                                           batch_size=self.batch_size,
                                           seq_len=self.max_char_len,
                                           max_lexicon_words_num=self.max_lexicon_words_num,
                                           word_length_tensor=word_length_tensor,
                                           dtype=tf.float32)

        initial_state = ner_lattice_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell=ner_lattice_lstm,
                                           inputs=[char_embed, lexicon_word_embed],
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        # projection:
        W = tf.get_variable("projection_w", [self.num_units, self.num_tags])
        b = tf.get_variable("projection_b", [self.num_tags])
        x_reshape = tf.reshape(outputs, [-1, self.num_units])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to time step
        outputs_project = tf.reshape(projection, [self.batch_size, -1, self.num_tags])


        seq_length = tf.convert_to_tensor(self.batch_size * [self.max_char_len], dtype=tf.int32)

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(outputs_project, labels, seq_length)

        # Add a training op to tune the parameters.
        #self.loss = tf.reduce_mean(-log_likelihood)
        self.loss = tf.reduce_sum(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.placeholders["char_ids"] = char_ids
        self.placeholders["lexicon_word_ids"] = lexicon_word_ids
        self.placeholders["word_length_tensor"] = word_length_tensor
        self.placeholders["labels"] = labels

    def train_model(self,data):
        init = tf.global_variables_initializer()
        sess = self.sess
        sess.run(init)

        iters = 100
        for iter in range(iters):
            random.shuffle(data.train_Ids)

            train_num = len(data.train_Ids)
            total_batch = train_num // self.batch_size
            for batch_id in range(total_batch):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                if end > train_num:
                    end = train_num
                instance = data.train_Ids[start:end]
                if not instance:
                    continue

                self.epoch += 1
                char_ids, lexicon_word_ids, word_length_tensor, labels = self.batchify_with_label(instance)

                #run模型
                feed_dict = {
                    self.placeholders["char_ids"]:char_ids,
                    self.placeholders["lexicon_word_ids"]:lexicon_word_ids,
                    self.placeholders["word_length_tensor"]:word_length_tensor,
                    self.placeholders["labels"]:labels,
                }

                _, losses = sess.run([self.train_op, self.loss],feed_dict=feed_dict)

                if self.epoch % 10 == 0:
                    print('*' * 100)
                    print(self.epoch, 'loss', losses)

    def load_data_and_embedding(self, data):
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = False
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False
        self.data_initialization(data, self.gaz_file, self.train_file, self.dev_file, self.test_file)
        data.generate_instance_with_gaz(self.train_file, 'train')
        data.generate_instance_with_gaz(self.dev_file, 'dev')
        data.generate_instance_with_gaz(self.test_file, 'test')
        data.build_word_pretrain_emb(self.char_emb)
        data.build_biword_pretrain_emb(self.bichar_emb)
        data.build_gaz_pretrain_emb(self.gaz_file)

    def data_initialization(self, data, gaz_file, train_file, dev_file, test_file):
        data.build_alphabet(train_file)
        data.build_alphabet(dev_file)
        data.build_alphabet(test_file)
        data.build_gaz_file(gaz_file)
        data.build_gaz_alphabet(train_file)
        data.build_gaz_alphabet(dev_file)
        data.build_gaz_alphabet(test_file)
        data.fix_alphabet()

    def batchify_with_label(self, input_batch_list):
        """
            input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
                words: word ids for one sentence. (batch_size, sent_len)
                chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            output:
                char_ids: (batch_size, )
                lexicon_word_ids: (batch_size, )
                word_length_tensor: (batch_size, )
                labels: (batch_size, )
        """
        batch_size = len(input_batch_list)
        chars_ids = [sent[0][0:self.max_char_len] for sent in input_batch_list]
        biwords = [sent[1][0:self.max_char_len] for sent in input_batch_list]
        chars_ids_split = [sent[2][0:self.max_char_len] for sent in input_batch_list]
        lexicon_words = [sent[3][0:self.max_char_len] for sent in input_batch_list]
        labels = [sent[4][0:self.max_char_len] for sent in input_batch_list]

        chars_ids = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), chars_ids))
        biwords = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), biwords))
        labels = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), labels))

        lexicon_word_ids = []
        word_length_tensor = []
        for sent in input_batch_list:
            lexicon_word_ids_sent = []
            word_length_tensor_sent = []
            for word_lexicon in sent[3][0:self.max_char_len]:
                word_lexicon_pad = list(map(lambda l: l + [0] * (self.max_lexicon_words_num - len(l)), word_lexicon))
                lexicon_word_ids_sent.append(word_lexicon_pad[0][0:self.max_lexicon_words_num])
                word_length_tensor_sent.append(word_lexicon_pad[1][0:self.max_lexicon_words_num])
            lexicon_word_ids.append(lexicon_word_ids_sent)
            word_length_tensor.append(word_length_tensor_sent)

        lexicon_word_ids = list(map(lambda l: l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)), lexicon_word_ids))
        word_length_tensor = list(map(lambda l: l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)), word_length_tensor))

        return chars_ids, lexicon_word_ids, word_length_tensor, labels

if __name__ == "__main__":
    confs = Configure(sys.argv[1])
    # model_save_path = "model/"
    # dev_ret_path = "tmp/"
    model = LatticeNet(conf=confs)
    model.data = Data()
    model.load_data_and_embedding(model.data)
    model.create_model()
    model.train_model(model.data)
    # model.save_parameters()
    # model.train_model()
