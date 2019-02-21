"""merge review to one sentence"""

import pickle
import numpy as np
import os

class MergeReview:
    def __init__(self):
        self.configs = {'train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/train_han_fasttext.pkl',
                        'testa_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/testa_han_fasttext.pkl',
                        'dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/dev_han_fasttext.pkl',
                        'new_train_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/merged_train_data.pkl',
                        'new_dev_data_path': '/datastore/liu121/sentidata2/data/aic2018_junyu/merged_dev_data.pkl'
                        }
        self.train_review, self.train_attr_label, self.train_senti_label, self.attribute_dic, self.word_dic, self.table = self.load_train_data()
        self.dev_review, self.dev_attr_label, self.dev_senti_label = self.load_dev_data()

    def load_train_data(self):
        assert os.path.exists(self.configs['train_data_path']) and os.path.getsize(self.configs['train_data_path']) > 0

        with open(self.configs['train_data_path'], 'rb') as f:
            attribute_dic, word_dic, train_labels, train_sentence, word_embed = pickle.load(f)

        train_attr_label = np.sum(np.reshape(train_labels, newshape=[-1, 20, 4])[:, :, 1:], axis=-1)
        train_senti_label = np.reshape(train_labels, newshape=[-1, 20, 4])[:, :, 1:]

        return train_sentence, train_attr_label, train_senti_label, attribute_dic, word_dic, word_embed

    def load_dev_data(self):
        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_labels, dev_review = pickle.load(f)

        dev_attr_label = np.sum(np.reshape(dev_labels, newshape=[-1, 20, 4])[:, :, 1:], axis=-1)
        dev_senti_label = np.reshape(dev_labels, newshape=[-1, 20, 4])[:, :, 1:]

        return dev_review, dev_attr_label, dev_senti_label

    def max_len(self,reviews):
        condition = np.equal(reviews,0)
        # (batch , review len, sentence len)
        reviews = np.where(condition,np.zeros_like(reviews),np.ones_like(reviews))
        return np.max(np.sum(np.sum(reviews,axis=2),axis=1))

    def merge(self,review):
        print('review shape: \n',np.shape(self.train_review))
        shape = np.shape(self.train_review)
        pad_id = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                for m in range(shape[2]):
                    pass

    def main(self):
        size1 = self.max_len(self.train_review)
        size2 = self.max_len(self.dev_review)
        print(size1)
        print(size2)

if __name__ == "__main__":
    mr = MergeReview()
    mr.main()