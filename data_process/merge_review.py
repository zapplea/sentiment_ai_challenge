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
        """
        1141
        :param reviews:
        :return:
        """
        condition = np.equal(reviews,0)
        # (batch , review len, sentence len)
        reviews = np.where(condition,np.zeros_like(reviews),np.ones_like(reviews))
        return np.max(np.sum(np.sum(reviews,axis=2),axis=1))

    def merge(self,reviews):
        print('review shape: \n',np.shape(reviews))
        shape = np.shape(reviews)
        merged_reviews = []
        pad_id = 0
        max_len = 1141
        for i in range(shape[0]):
            new_review = []
            for j in range(shape[1]):
                sentence = reviews[i][j]
                condition = np.not_equal(sentence,pad_id)
                new_review+= sentence[condition].tolist()
            if len(new_review)<max_len:
                pad_len = max_len - len(new_review)
                new_review+=np.zeros(shape=(pad_len,)).tolist()
            merged_reviews.append(new_review)

        merged_reviews = np.array(merged_reviews).astype('int32')
        return merged_reviews

    def main(self):
        merged_train_review = self.merge(self.train_review)
        print('merged_train_review: ',np.shape(merged_train_review))

        merged_dev_review = self.merge(self.dev_review)
        print('merged_dev_review: ',np.shape(merged_dev_review))

if __name__ == "__main__":
    mr = MergeReview()
    mr.main()