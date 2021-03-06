"""merge review to one sentence"""

import pickle
import numpy as np
import os

class MergeReview:
    def __init__(self,config):
        self.configs = config
        self.train_review, self.train_attr_label, self.train_senti_label, self.attribute_dic, self.word_dic, self.table = self.load_train_data()
        self.dev_review, self.dev_attr_label, self.dev_senti_label = self.load_dev_data()

    def load_tecentWordsVec(self):
        with open(self.configs['tencent_wordsVec_path'],'rb') as f:
            dic = pickle.load(f)
            tenc_id2word_dic = dic['id2word']
            tenc_word2id_dic = dic['word2id']
            tenc_wordsVec = dic['wordsVec']
        fasttext_id2word_dic = {}
        for key in self.word_dic:
            fasttext_id2word_dic[self.word_dic[key]] = key

        new_wordsVec = []
        new_word_dic = {}
        for i in range(len(fasttext_id2word_dic)):
            word = fasttext_id2word_dic[i]
            if word in tenc_word2id_dic:
                tenc_id = tenc_word2id_dic[word]
                new_wordsVec.append(tenc_wordsVec[tenc_id])
            else:
                new_wordsVec.append(self.table[i][:200])
            new_word_dic[word]=i
        return new_word_dic, np.array(new_wordsVec).astype('float32')

    def load_train_data(self):
        assert os.path.exists(self.configs['train_data_path']) and os.path.getsize(self.configs['train_data_path']) > 0

        with open(self.configs['train_data_path'], 'rb') as f:
            attribute_dic, word_dic, train_labels, train_sentence, word_embed = pickle.load(f)

        train_attr_label = np.sum(np.reshape(train_labels, newshape=[-1, 20, 4])[:,:,1:], axis=-1)
        train_senti_label = np.reshape(train_labels, newshape=[-1, 20, 4])[:,:,1:]

        return train_sentence, train_attr_label, train_senti_label, attribute_dic, word_dic, word_embed

    def load_dev_data(self):
        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_labels, dev_review = pickle.load(f)

        dev_attr_label = np.sum(np.reshape(dev_labels, newshape=[-1, 20, 4])[:,:,1:], axis=-1)
        dev_senti_label = np.reshape(dev_labels, newshape=[-1, 20, 4])[:,:,1:]

        return dev_review, dev_attr_label, dev_senti_label

    def max_len(self,reviews):
        """
        1141
        :param reviews:
        :return:
        """
        condition = np.equal(reviews,0)
        # (batch , review len, sentence len)
        reviews = np.where(condition,np.zeros_like(reviews),np.ones_like(reviews))
        return np.max(np.sum(np.sum(reviews,axis=2),axis=1))


    def main(self):
        new_word_dic, new_wordsVec = self.load_tecentWordsVec()
        with open(self.configs['tenc_train_data_path'],'wb') as f:
            pickle.dump((self.train_review[:self.configs['up']], self.train_attr_label[:self.configs['up']], self.train_senti_label[:self.configs['up']], self.attribute_dic, new_word_dic, new_wordsVec),f,protocol=4)
        with open(self.configs['tenc_dev_data_path'],'wb') as f:
            pickle.dump((self.dev_review[:self.configs['up']],self.dev_attr_label[:self.configs['up']], self.dev_senti_label[:self.configs['up']]),f,protocol=4)

if __name__ == "__main__":
    print('run program: ')
    config = {'train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/train_han_fasttext.pkl',
              'testa_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/testa_han_fasttext.pkl',
              'dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/dev_han_fasttext.pkl',
              'tenc_train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_train_data.pkl',
              'tenc_dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_dev_data.pkl',
              'tencent_wordsVec_path':'/datastore/liu121/wordEmb/tencent_cn/tencent_wordsVec.pkl',
              'up':None
             }
    mr = MergeReview(config)
    mr.main()