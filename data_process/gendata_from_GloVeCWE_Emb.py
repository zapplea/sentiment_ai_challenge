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
        # Fixed: for padded words, it also needs char
        for j in range(len(review_collection)):
            review = review_collection[j]
            reviewID_ls = []
            reviewCharID_ls = []
            for sentence in review:
                sentenceID_ls = []
                sentenceCharID_ls = []
                sentence = sentence.split(' ')
                for i in range(len(sentence)):
                    word = sentence[i]

                    # Word
                    if len(list(word)) > config['corpus']['max_word_len']:
                        sentenceID_ls.append(self.word_to_id[self.config['corpus']['unknown_word']])
                    else:
                        sentenceID_ls.append(self.word_to_id[word])
                    if len(sentenceID_ls)<self.config['corpus']['max_sentence_len']:
                        dim0=self.config['corpus']['max_sentence_len']-len(sentenceID_ls)
                        patch = (np.ones(shape=(dim0,))*self.word_to_id[self.config['corpus']['padding_word']]).tolist()
                        sentenceID_ls.extend(patch)

                    # Char
                    if len(list(word)) > self.config['corpus']['max_word_len']:
                        sentenceCharID_ls.append((np.ones(shape=(self.config['corpus']['max_word_len'],),dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                    else:
                        charID_ls = []
                        char_ls = list(word)
                        for char in char_ls:
                            charID_ls.append(self.char_to_id[char])
                        # pad char list
                        if len(charID_ls)<self.config['corpus']['max_word_len']:
                            shape = (self.config['corpus']['max_word_len'] - len(charID_ls),)
                            charID_ls.extend((np.ones(shape = shape,dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                        sentenceCharID_ls.append(charID_ls)
                # pad sentence char list
                if len(sentenceCharID_ls)<self.config['corpus']['max_sentence_len']:
                    dim0 = self.config['corpus']['max_sentence_len'] - len(sentenceCharID_ls)
                    dim1 = self.config['corpus']['max_word_len']
                    sentenceCharID_ls.extend((np.ones(shape=(dim0,dim1),dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                reviewCharID_ls.append(sentenceCharID_ls)
                reviewID_ls.append(sentenceID_ls)
            # pad review char list
            # pad review list
            if len(reviewCharID_ls)<self.config['corpus']['max_review_len']:
                dim0 = self.config['corpus']['max_review_len'] - len(reviewCharID_ls)
                dim1 = self.config['corpus']['max_sentence_len']
                dim2 = self.config['corpus']['max_word_len']
                reviewCharID_ls.extend((np.ones(shape=(dim0,dim1,dim2),dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                reviewID_ls.extend((np.ones(shape=(dim0,dim1,),dtype='int32')*self.word_to_id[self.config['corpus']['padding_word']]).tolist())

            prepared_review_collection.append(reviewID_ls)
            prepared_char_collection.append(reviewCharID_ls)

            label = label_collection[j]
            attr_label_ls = []
            senti_label_ls = []
            for i in range(len(label)):
                if label[i] in {1,0,-1}:
                    attr_label_ls.append(1)
                    if label[i]==1:
                        senti_label_ls.append([1,0,0])
                    elif label[i] == 0:
                        senti_label_ls.append([0,1,0])
                    else:
                        senti_label_ls.append([0,0,1])
                else:
                    attr_label_ls.append(0)
                    senti_label_ls.append([0,0,0])
            prepared_attr_label_collection.append(attr_label_ls)
            prepared_senti_label_collection.append(senti_label_ls)
        return prepared_review_collection,prepared_attr_label_collection,prepared_senti_label_collection,prepared_char_collection

    def merged_read_corpus(self,fname):
        data = pd.read_pickle(fname)
        label_collection = data[:, 2]
        review_collection = data[:, 1]
        prepared_review_collection = []
        prepared_attr_label_collection = []
        prepared_senti_label_collection = []
        prepared_char_collection = []
        # Fixed: for padded words, it also needs char
        for j in range(len(review_collection)):
            review = review_collection[j]
            reviewID_ls = []
            reviewCharID_ls = []
            for sentence in review:
                sentence = sentence.split(' ')
                for i in range(len(sentence)):
                    word = sentence[i]

                    # Word
                    if len(list(word)) > config['corpus']['max_word_len']:
                        reviewID_ls.append(self.word_to_id[self.config['corpus']['unknown_word']])
                    else:
                        reviewID_ls.append(self.word_to_id[word])

                    # Char
                    if len(list(word)) > self.config['corpus']['max_word_len']:
                        reviewCharID_ls.append((np.ones(shape=(self.config['corpus']['max_word_len'],), dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                    else:
                        charID_ls = []
                        char_ls = list(word)
                        for char in char_ls:
                            charID_ls.append(self.char_to_id[char])
                        # pad char list
                        if len(charID_ls) < self.config['corpus']['max_word_len']:
                            shape = (self.config['corpus']['max_word_len'] - len(charID_ls),)
                            charID_ls.extend((np.ones(shape=shape, dtype='int32') * self.char_to_id[self.config['corpus']['padding_char']]).tolist())
                        reviewCharID_ls.append(charID_ls)
            if len(reviewID_ls)<self.config['corpus']['max_review_len']*self.config['corpus']['max_sentence_len']:
                dim0 = self.config['corpus']['max_review_len']*self.config['corpus']['max_sentence_len'] - len(reviewID_ls)
                dim1 = self.config['corpus']['max_word_len']
                reviewID_ls.extend((np.ones(shape=(dim0,),dtype='int32')*self.word_to_id[self.config['corpus']['padding_word']]).tolist())
                reviewCharID_ls.extend((np.ones(shape=(dim0,dim1),dtype='int32')*self.char_to_id[self.config['corpus']['padding_char']]).tolist())
            prepared_review_collection.append(reviewID_ls)
            prepared_char_collection.append(reviewCharID_ls)

            label = label_collection[j]
            attr_label_ls = []
            senti_label_ls = []
            for i in range(len(label)):
                if label[i] in {1, 0, -1}:
                    attr_label_ls.append(1)
                    if label[i] == 1:
                        senti_label_ls.append([1, 0, 0])
                    elif label[i] == 0:
                        senti_label_ls.append([0, 1, 0])
                    else:
                        senti_label_ls.append([0, 0, 1])
                else:
                    attr_label_ls.append(0)
                    senti_label_ls.append([0, 0, 0])
            prepared_attr_label_collection.append(attr_label_ls)
            prepared_senti_label_collection.append(senti_label_ls)
        return prepared_review_collection, prepared_attr_label_collection, prepared_senti_label_collection, prepared_char_collection

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
    @staticmethod
    def split(data,config):
        new_data=[]
        for review in data:
            new_review = []
            for sentence in new_review:
                sentence = sentence.split(' ')
                if len(sentence)>config['corpus']['max_sentence_len']:
                    multiple = len(sentence)//config['corpus']['max_sentence_len']
                    mod = len(sentence)%config['corpus']['max_sentence_len']
                    if mod == 0:
                        rng = multiple
                    else:
                        rng = multiple+1
                    for i in range(rng):
                        start = i*config['corpus']['max_sentence_len']
                        stop = start+config['corpus']['max_sentence_len']-1
                        new_review.append(' '.join(sentence[start:stop]))
                else:
                    new_review.append(' '.join(sentence))
        new_data.append(new_review)
        return new_data

    @staticmethod
    def stats(config):
        freq_train={range(0,200):0,
              range(200,300):0,
              range(300,400):0,
              range(400,500):0,
              range(500,600):0,
              range(600,700):0,
              range(700,800):0,
              range(800,900):0}
        freq_val = {range(0, 200): 0,
                range(200, 300): 0,
                range(300, 400): 0,
                range(400, 500): 0,
                range(500, 600): 0,
                range(600, 700): 0,
                range(700, 800): 0,
                range(800, 900): 0}
        fname = config['corpus']['train_path']
        train_data = pd.read_pickle(fname)
        review_collection = train_data[:, 1]
        max_review = None
        max_sentence = None
        max_review_len = 0
        max_sent_len = 0
        second_sent_len =0
        for review in review_collection:
            if len(review) > max_review_len:
                max_review_len=len(review)

            for sentence in review:
                sentence=sentence.split(' ')
                for key in freq_train:
                    if len(sentence) in key:
                        freq_train[key]+=1
                        break
                if len(sentence)>max_sent_len:
                    max_sent_len=len(sentence)
                else:
                    if len(sentence)>second_sent_len:
                        second_sent_len=len(sentence)

        fname = config['corpus']['val_path']
        dev_data = pd.read_pickle(fname)
        review_collection = dev_data[:, 1]
        for review in review_collection:
            if len(review) > max_review_len:
                max_review_len=len(review)
            for sentence in review:
                sentence=sentence.split(' ')
                for key in freq_val:
                    if len(sentence) in key:
                        freq_val[key]+=1
                        break
                if len(sentence)>max_sent_len:
                    max_sent_len=len(sentence)
                else:
                    if len(sentence)>second_sent_len:
                        second_sent_len=len(sentence)
        print('max review len: ',max_review_len)
        print('max sent len: ',max_sent_len)
        print('second sent len: ',second_sent_len)
        print('freq_train: ',freq_train)
        print('freq_val: ',freq_val)



if __name__ == "__main__":
    config = {'corpus':{'train_path':'/datastore/liu121/sentidata2/data/meituan_jieba/train_cut.pkl',
                        'val_path':'/datastore/liu121/sentidata2/data/meituan_jieba/val_cut.pkl',
                        'max_word_len':11,
                        'unknown_word':'#UNK#',
                        'padding_word':'#PAD#',
                        'padding_char':'#PAD#',
                        'max_sentence_len':200,
                        'max_review_len':None,
                        'old_train_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl'},
              'emb':{'wordEmb_path':'/datastore/liu121/wordEmb/aic2018cwe_wordEmb.pkl',
                     'charEmb_path':'/datastore/liu121/charEmb/aic2018cwe_charEmb.pkl'},
              'training_data':{'train_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_train_cwe.pkl',
                               'dev_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_dev_cwe.pkl'}}
    GenDataGloVeCWE.stats(config)
    # gen=GenDataGloVeCWE(config)
    # gen.prepare_data()