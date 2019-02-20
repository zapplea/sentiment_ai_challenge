import numpy as np
import pickle
import os
import time

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_review, self.train_attr_label, self.train_senti_label, self.attribute_dic, self.word_dic, self.table = self.load_train_data()
        self.vocab = list(self.word_dic.keys())
        self.train_data_size = len(self.train_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Training Data')

        self.dev_review, self.dev_attr_label, self.dev_senti_label = self.load_dev_data()
        self.dev_data_size = len(self.dev_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Dev Data')

        self.test_review = self.load_test_data()
        self.test_data_size = len(self.test_review)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Test Data')

    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_review, self.train_attr_label, self.train_senti_label, _ = self.unison_shuffled(self.train_review,
                                                                             self.train_attr_label, self.train_senti_label)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Shuffling Data:')

        if start < end:
            batches_review = self.train_review[start:end]
            batches_attr_label = self.train_attr_label[start:end]
            batches_senti_label = self.train_senti_label[start:end]
        else:
            batches_review = self.train_review[train_size - self.configs['batch_size']:train_size]
            batches_attr_label = self.train_attr_label[train_size - self.configs['batch_size']:train_size]
            batches_senti_label = self.train_senti_label[train_size - self.configs['batch_size']:train_size]


        return batches_review, batches_attr_label, batches_senti_label

    def dev_data_generator(self, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch.
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """

        dev_size = self.dev_data_size
        start = batch_num * self.configs['batch_size'] % dev_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % dev_size
        if start < end:
            batches_review = self.dev_review[start:end]
            batches_attr_label = self.dev_attr_label[start:end]
            batches_senti_label = self.dev_senti_label[start:end]
        else:
            batches_review = self.dev_review[start:]
            batches_attr_label = self.dev_attr_label[start:]
            batches_senti_label = self.dev_senti_label[start:]

        return batches_review, batches_attr_label, batches_senti_label


    def table_generator(self):

        if self.configs['word_emb_init'] is not None:
            with open(self.configs['word_emb_init'], 'rb') as f:
                self._word_embedding_init = pickle.load(f, encoding='latin1')
        else:
            self._word_embedding_init = np.random.random(size=[self.configs['vocab_size'], self.configs['emb_size']])

        return self._word_embedding_init


    def unison_shuffled(self, a, b, c):
        np.random.seed(self.configs['rand_seed'])
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], p

    def unison_shuffled_copies(self, a, b, c, d):
        assert len(a) == len(b) == len(c) == len(d)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p], p

    def shuffle_response(self, candidate_id, response, response_len, label):
        """
        responses contain ground truth id
        :param response: (batch_size, options_num, max_turn_len)
        :param response_len: (batch_size, options_num)
        :param label: (batch_size, options_num)
        :return:
        """
        candidate_id, response, response_len, label = list(candidate_id), list(response), list(response_len), list(label)
        for i in range(len(response)):
            candidate_id[i],response[i], response_len[i], label[i], _ = self.unison_shuffled_copies(
                candidate_id[i],response[i],response_len[i],label[i])

        return candidate_id, response, response_len, label


    def get_context(self, context):
        """

        :param data:
        :param eos_idx:
        :param max_turn_num:
        :param max_turn_len:
        :return: array of tuple, tuple:(sent_list, example_turn_num, example_turn_len)
        """

        eos_idx = self.configs['_EOS_']
        max_turn_num = self.configs['max_turn_num']
        max_turn_len = self.configs['max_turn_len']
        context.reset_index(drop=True, inplace=True)

        saver = []

        for c in range(context.shape[0]):

            example_turn_len = []

            # spilt to sentence and padding 0 to every sentence
            sent_list = []
            tmp = []
            num = 0
            for word in context['c'][c]:
                if word != eos_idx:
                    num += 1
                    tmp.append(word)
                else:
                    if num >= max_turn_len:
                        tmp = tmp[0:max_turn_len]
                        example_turn_len.append(max_turn_len)
                    else:
                        pad = [0] * (max_turn_len - num)
                        example_turn_len.append(num)
                        tmp += pad
                    sent_list.append(np.array(tmp,dtype=np.int32))
                    tmp = []
                    num = 0

            # padding zero vector to normalize turn num
            pad_sent = np.array([0] * max_turn_len,dtype=np.int32)
            if len(sent_list) < max_turn_num:
                example_turn_num = len(sent_list)
                for i in range(max_turn_num - len(sent_list)):
                    sent_list.append(pad_sent)
                    example_turn_len.append(0)
            else:
                example_turn_num = max_turn_num
                sent_list = sent_list[-max_turn_num:]
                example_turn_len = example_turn_len[-max_turn_num:]

            res = np.array([context['example-id'][c], np.array(sent_list,dtype=np.int32), example_turn_num,
                            np.array(example_turn_len,dtype=np.int32)],dtype=object)
            saver.append(res)

        return np.array(saver)


    def get_response(self, response, flag='test'):
        """

        :param PATH:
        :return: array of tuple, tuple:(sent, example_response_len)
        """

        assert flag in ['test','train','dev']

        max_respone_len = self.configs['max_turn_len']
        options_num = self.configs['options_num']
        response.reset_index(drop=True, inplace=True)
        saver = []

        for e in range(int(response.shape[0] / options_num)):
            example_data  = response[e*options_num:(e+1)*options_num]

            assert len(set(example_data['example-id'])) == 1

            example_data = example_data.reset_index(drop=True)
            example_sent = []
            example_response_len = []
            for r in range(options_num):
                options_response_len = 0
                options_sent = []
                for word in example_data['r'][r]:
                    if options_response_len >= max_respone_len:
                        break
                    else:
                        options_sent.append(word)
                        options_response_len += 1

                if len(options_sent) < max_respone_len:
                    pad = [0] * (max_respone_len - len(options_sent))
                    options_sent += pad

                example_sent.append(np.array(options_sent,dtype=np.int32))
                example_response_len.append(options_response_len)

            res = np.array([response['example-id'][e * options_num], np.array(example_data['candidate-id']),
                            np.array(example_sent,dtype=np.int32), np.array(example_response_len,dtype=np.int32),
                            np.array(example_data['y'])],
                           dtype=object)
            saver.append(res)

        return np.array(saver)

    def generate_bert_sent_emb(self, revs):

        def id2w(id):
            for i in id:
                if i != self.configs['_EOS_']:
                    yield self.vocab[i]

        def sent2emb(rev):
            return self.bc.encode(rev)

        # batch_rev_emb = []
        # for rev in revs:
        #     rev_str = []
        #     for sent in rev:
        #         tmp = ''.join(list(id2w(sent)))
        #         if tmp:
        #             rev_str.append(tmp)
        #         else:
        #             rev_str.append('。')
        #     batch_rev_emb.append(sent2emb(rev_str))

        rev_str = []
        for rev in revs:
            for sent in rev:
                tmp = ''.join(list(id2w(sent)))
                if tmp:
                    rev_str.append(tmp)
                else:
                    rev_str.append('。')

        batch_rev_emb = sent2emb(rev_str)
        batch_rev_emb = np.reshape(batch_rev_emb, newshape=[-1, self.configs['max_rev_len'], self.configs['bert_dim']])

        return batch_rev_emb

    def load_train_data(self):

        assert os.path.exists(self.configs['train_data_path']) and os.path.getsize(self.configs['train_data_path']) > 0

        with open(self.configs['train_data_path'], 'rb') as f:
            train_review, train_attr_label, _, train_senti_label, attribute_dic, word_dic, table = pickle.load(f)

        return train_review, train_attr_label, train_senti_label, attribute_dic, word_dic, table

    def load_dev_data(self):

        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_review, dev_attr_label, _, dev_senti_label = pickle.load(f)

        return dev_review, dev_attr_label, dev_senti_label

    def load_test_data(self):

        assert os.path.exists(self.configs['testa_data_path']) and os.path.getsize(self.configs['testa_data_path']) > 0

        with open(self.configs['testa_data_path'], 'rb') as f:
            testa_review = pickle.load(f)

        return testa_review