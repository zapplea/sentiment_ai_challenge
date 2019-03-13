from os.path import join
import pickle
import pandas as pd
import numpy as np

# TODO: cwep and cwe are different.
class GenDataGloVeCWE:
    def __init__(self,config):
        self.config = config
        self.load_word_embeddings()
        self.load_char_embeddings()
        self.load_training_data()
        self.load_validation_data()
        self.load_attr_dic()

    def load_word_embeddings(self):
        with open(self.config['emb']['wordEmb_path'],'rb') as f:
            dic = pickle.load(f)
            self.word_to_id = dic['word_to_id']
            self.word_embeddings = dic['word_embeddings']

    def load_char_embeddings(self):
        with open(self.config['emb']['charEmb_path'],'rb') as f:
            dic = pickle.load(f)
            self.char_to_id = dic['char_to_id']
            self.char_embeddings = dic['char_embeddings']

    def read_corpus(self,fname):
        data = pd.read_pickle(fname)
        label_collection = data[:,2]
        review_collection = data[:,1]
        prepared_review_collection = []
        prepared_attr_label_collection = []
        prepared_senti_label_collection = []
        prepared_char_collection = []
        for i in range(len(review_collection)):
            review = review_collection[i]
            reviewID_ls = []
            reviewCharID_ls = []
            for sentence in review:
                sentenceID_ls = []
                sentenceCharID_ls = []
                sentence = sentence.split(' ')
                for i in range(len(sentence)):
                    word = sentence[i]
                    if len(list(word)) > config['corpus']['max_word_len']:
                        sentenceID_ls.append(self.word_to_id[self.config['corpus']['unknown_word']])
                    else:
                        sentenceID_ls.append(self.word_to_id[word])
                    if len(sentenceID_ls)<self.config['corpus']['max_sentence_len']:
                        dim0=self.config['corpus']['max_sentence_len']-len(sentenceID_ls)
                        patch = (np.ones(shape=(dim0,))*self.word_to_id[self.config['corpus']['padding_word']]).tolist()
                        sentenceID_ls.extend(patch)
                    if word == self.config['corpus']['unknown_word']:
                        sentenceCharID_ls.append(np.zeros(shape=(self.config['corpus']['max_word_len'],),dtype='int32').tolist())
                    else:
                        charID_ls = []
                        char_ls = list(word)
                        for char in char_ls:
                            charID_ls.append(self.char_to_id[char])
                        if len(charID_ls)<self.config['corpus']['max_word_len']:
                            shape = (self.config['corpus']['max_word_len'] - len(charID_ls),)
                            charID_ls.extend(np.zeros(shape = shape,dtype='int32').tolist())
                        sentenceCharID_ls.append(charID_ls)
                reviewCharID_ls.append(sentenceCharID_ls)
                reviewID_ls.append(sentenceID_ls)
            prepared_review_collection.append(reviewID_ls)
            prepared_char_collection.append(reviewCharID_ls)

            label = label_collection[i]
            attr_label_ls = []
            senti_label_ls = []
            for j in range(len(label)):
                if label[j] in {1,0,-1}:
                    attr_label_ls.append(1)
                    if label[j]==1:
                        senti_label_ls.append([1,0,0])
                    elif label[j] == 0:
                        senti_label_ls.append([0,1,0])
                    else:
                        senti_label_ls.append([0,0,1])
                else:
                    attr_label_ls.append(0)
                    senti_label_ls.append([0,0,0])
            prepared_attr_label_collection.append(attr_label_ls)
            prepared_senti_label_collection.append(senti_label_ls)
        return prepared_review_collection,prepared_attr_label_collection,prepared_senti_label_collection,prepared_char_collection


    def load_training_data(self):
        self.train_review_collection,self.train_attr_label_collection,self.train_senti_label_collection,self.train_char_collection= \
            self.read_corpus(self.config['corpus']['train_path'])

    def load_validation_data(self):
        self.val_review_collection,self.val_attr_label_collection, self.val_senti_label_collection, self.val_char_collection = \
            self.read_corpus(self.config['corpus']['val_path'])

    def load_attr_dic(self):
        with open(self.config['corpus']['old_train_data_path'],'rb') as f:
            _, _, _, _, self.attribute_dic, _, _, _ = pickle.load(f)

    def write(self,fname,data):
        with open(fname,'wb') as f:
            pickle.dump(data,f,protocol=4)

    def prepare_data(self,):
        training_data = (self.train_review_collection,self.train_char_collection,self.train_attr_label_collection,
                         self.train_senti_label_collection,self.attribute_dic,self.word_to_id,self.word_embeddings,self.char_embeddings)
        self.write(self.config['training_data']['train_path'],training_data)

        dev_data = (self.val_review_collection,self.val_char_collection,self.val_attr_label_collection,
                    self.val_senti_label_collection)
        self.write(self.config['training_data']['dev_path'],dev_data)

    def stats(self):
        fname = self.config['corpus']['train_path']
        train_data = pd.read_pickle(fname)
        review_collection = train_data[:, 1]
        max_review_len = 0
        max_sent_len = 0
        for review in review_collection:
            if len(review) > max_review_len:
                max_review_len=len(review)
            for sentence in review:
                sentence=sentence.split(' ')
                if len(sentence)>max_sent_len:
                    max_sent_len=len(sentence)

        fname = self.config['corpus']['dev_path']
        dev_data = pd.read_pickle(fname)
        review_collection = train_data[:, 1]
        for review in review_collection:
            if len(review) > max_review_len:
                max_review_len=len(review)
            for sentence in review:
                sentence=sentence.split(' ')
                if len(sentence)>max_sent_len:
                    max_sent_len=len(sentence)
        print('max review len: ',max_review_len)
        print('max sent len: ',max_sent_len)


if __name__ == "__main__":
    config = {'corpus':{'train_path':'/datastore/liu121/sentidata2/data/meituan_jieba/train_cut.pkl',
                        'val_path':'/datastore/liu121/sentidata2/data/meituan_jieba/val_cut.pkl',
                        'max_word_len':11,
                        'unknown_word':'#UNK#',
                        'padding_word':'#PAD#',
                        'padding_char':'#PAD#',
                        'max_sentence_len':None,
                        'max_review_len':None,
                        'old_train_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl'},
              'emb':{'wordEmb_path':'',
                     'charEmb_path':''},
              'training_data':{'train_path':'',
                               'dev_path':''}}
    gen=GenDataGloVeCWE(config)
    gen.stats()
    # gen.prepare_data()