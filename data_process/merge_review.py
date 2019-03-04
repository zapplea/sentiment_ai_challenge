"""merge review to one sentence"""

import pickle
import numpy as np
import os
import re

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

        train_attr_label = np.sum(np.reshape(train_labels, newshape=[-1, 20, 4]), axis=-1)
        train_senti_label = np.reshape(train_labels, newshape=[-1, 20, 4])

        return train_sentence, train_attr_label, train_senti_label, attribute_dic, word_dic, word_embed

    def load_dev_data(self):
        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_labels, dev_review = pickle.load(f)

        dev_attr_label = np.sum(np.reshape(dev_labels, newshape=[-1, 20, 4]), axis=-1)
        dev_senti_label = np.reshape(dev_labels, newshape=[-1, 20, 4])

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

    def merge(self,reviews,word_dic):
        print('review shape: \n',np.shape(reviews))
        id_to_word = {}
        for word in word_dic:
            id_to_word[word_dic[word]] = word
        shape = np.shape(reviews)
        merged_reviews = []
        pad_id = 0
        max_len = 1141
        for i in range(shape[0]):
            new_review = []
            for j in range(shape[1]):
                sentence = reviews[i][j]
                condition = np.not_equal(sentence,pad_id)
                # new_review+= sentence[condition].tolist()
                for word_id in sentence[condition].tolist():
                    if re.search(u'[\u4e00-\u9fff]',id_to_word[word_id]):
                        new_review.append(word_id)
            if len(new_review)<max_len:
                pad_len = max_len - len(new_review)
                new_review+=np.zeros(shape=(pad_len,)).tolist()
            merged_reviews.append(new_review)

        merged_reviews = np.array(merged_reviews).astype('int32')
        return merged_reviews

    def load_charEmb(self):
        with open(self.configs['charEmb_path']) as f:
            char_ls = ['#PAD#',]
            vec_ls = [np.zeros(shape=(50,),dtype='float32'),]
            for line in f:
                line=line.replace('\n','')
                ls = line.split(' ')
                char_ls.append(ls[0])
                vec_ls.append(np.array(list(map(float,ls[1:]))).astype('float32'))

        return char_ls, np.array(vec_ls).astype('float32')

    def doc_to_char(self,review,word_dic,char_ls,char_vecs):
        """

        :param review: (batch, review len)
        :return:
        """
        # TODO: use np.zeros() to reduce time complexity
        id_to_word = {}
        for word in word_dic:
            id_to_word[word_dic[word]] = word
        allreviews_char_id_ls = []
        max_char = 11
        for i in range(review.shape[0]):
            review_char_id_ls = []
            for j in range(review.shape[1]):
                word_id = review[i][j]
                word = id_to_word[word_id]
                if word != "#PAD#":
                    chars = list(word)
                    word_char_id_ls = []
                    for char in chars:
                        if char not in set(char_ls):
                            char_ls.append(char)
                            char_vecs = np.concatenate([char_vecs,np.random.uniform(size=(1,50)).astype('float32')],axis=0)
                        char_id = char_ls.index(char)
                        word_char_id_ls.append(char_id)
                    if len(word_char_id_ls)<max_char:
                        word_char_id_ls+=np.zeros(shape=(max_char-len(word_char_id_ls),)).astype('int32').tolist()
                else:
                    word_char_id_ls = np.zeros(shape=(max_char,),dtype='int32').tolist()
                review_char_id_ls.append(word_char_id_ls)
            print(review_char_id_ls)
            exit()
            allreviews_char_id_ls.append(review_char_id_ls)
        return np.array(allreviews_char_id_ls).astype('int32'), char_ls,char_vecs

    def measure_char_len(self,review,word_dic):
        """

        :param review: (batch, review len)
        :return:
        """
        id_to_word = {}
        for word in word_dic:
            id_to_word[word_dic[word]] = word
        allreviews_char_id_ls = []
        max_char = 10
        extra_word_freq = {}
        for i in range(review.shape[0]):
            review_char_id_ls = []
            for j in range(review.shape[1]):
                word_id = review[i][j]
                word = id_to_word[word_id]
                if word != "#PAD#":
                    chars = list(word)
                    if len(chars)>max_char:
                        if len(chars) in extra_word_freq:
                            extra_word_freq[len(chars)].append(word)
                        else:
                            extra_word_freq[len(chars)]=[word,]
        count = 0
        for key in extra_word_freq:
            word_ls = extra_word_freq[key]
            print(key)
            print(len(word_ls))
            print(word_ls)
            print('==================================')
        exit()


    def main(self):
        print('load charEmb')
        char_ls, char_vecs = self.load_charEmb()
        print('load tenc word vec')
        new_word_dic, new_wordsVec = self.load_tecentWordsVec()
        print('merge')
        merged_train_review = self.merge(self.train_review,new_word_dic)
        print('merged_train_review: ', np.shape(merged_train_review))
        merged_dev_review = self.merge(self.dev_review,new_word_dic)
        print('merged_dev_review: ', np.shape(merged_dev_review))

        if not self.configs['is_charEmb']:
            with open(self.configs['merged_train_data_path'],'wb') as f:
                pickle.dump((merged_train_review[:self.configs['up']], self.train_attr_label[:self.configs['up']], self.train_senti_label[:self.configs['up']], self.attribute_dic, new_word_dic, new_wordsVec),f,protocol=4)
            with open(self.configs['merged_dev_data_path'],'wb') as f:
                pickle.dump((merged_dev_review[:self.configs['up']],self.dev_attr_label[:self.configs['up']], self.dev_senti_label[:self.configs['up']]),f,protocol=4)
        else:
            # self.measure_char_len(review=merged_train_review,word_dic=new_word_dic)
            # self.measure_char_len(review=merged_dev_review,word_dic=new_word_dic)
            print('train char')
            merged_train_char,char_ls,char_vecs = self.doc_to_char(review=merged_train_review,word_dic=new_word_dic,char_ls=char_ls,char_vecs = char_vecs)
            print('train char.shape: ',merged_train_char.shape)
            print('dev char')
            merged_dev_char,char_ls,char_vecs = self.doc_to_char(review=merged_dev_review,word_dic=new_word_dic,char_ls=char_ls,char_vecs = char_vecs)
            print('dev char.shape: ',merged_dev_char.shape)
            with open(self.configs['merged_train_data_path'],'wb') as f:
                pickle.dump((merged_train_review[:self.configs['up']],merged_train_char[:self.configs['up']], self.train_attr_label[:self.configs['up']], self.train_senti_label[:self.configs['up']], self.attribute_dic, new_word_dic, new_wordsVec,char_vecs),f,protocol=4)
            with open(self.configs['merged_dev_data_path'],'wb') as f:
                pickle.dump((merged_dev_review[:self.configs['up']],merged_dev_char[:self.configs['up']],self.dev_attr_label[:self.configs['up']], self.dev_senti_label[:self.configs['up']]),f,protocol=4)

if __name__ == "__main__":
    print('run program: ')
    config = {'train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/train_han_fasttext.pkl',
              'testa_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/testa_han_fasttext.pkl',
              'dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/dev_han_fasttext.pkl',
              'merged_train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl',
              'merged_dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_dev_data_withChar.pkl',
              'tencent_wordsVec_path':'/datastore/liu121/wordEmb/tencent_cn/tencent_wordsVec.pkl',
              'charEmb_path':'/datastore/liu121/charEmb/aic2018_glove_charVec.txt',
              'is_charEmb':True,
              'up':None,
             }
    mr = MergeReview(config)
    mr.main()