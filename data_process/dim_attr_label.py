import pickle
import os
import numpy as np

class DimAttrLabel:
    def __init__(self):
        self.configs={'train_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/train_han_fasttext.pkl',
                      'testa_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/testa_han_fasttext.pkl',
                      'dev_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/dev_han_fasttext.pkl'
                      }
        self.train_review, self.train_attr_label, self.train_senti_label, self.attribute_dic, self.word_dic, self.table = self.load_train_data()
        self.dev_review, self.dev_attr_label, self.dev_senti_label = self.load_dev_data()
        self.test_review = self.load_test_data()

    def load_train_data(self):

        assert os.path.exists(self.configs['train_data_path']) and os.path.getsize(self.configs['train_data_path']) > 0

        with open(self.configs['train_data_path'], 'rb') as f:
            attribute_dic, word_dic, train_labels, train_sentence, word_embed = pickle.load(f)

        train_attr_label = np.sum(np.reshape(train_labels, newshape=[-1,20,4])[:,:,1:], axis=-1)
        train_senti_label = np.reshape(train_labels, newshape=[-1, 20, 4])[:, :, 1:]

        return train_sentence, train_attr_label, train_senti_label, attribute_dic, word_dic, word_embed

    def load_dev_data(self):

        assert os.path.exists(self.configs['dev_data_path']) and os.path.getsize(self.configs['dev_data_path']) > 0

        with open(self.configs['dev_data_path'], 'rb') as f:
            dev_labels, dev_review = pickle.load(f)

        dev_attr_label = np.sum(np.reshape(dev_labels, newshape=[-1, 20, 4])[:, :, 1:], axis=-1)
        dev_senti_label = np.reshape(dev_labels, newshape=[-1, 20, 4])[:, :, 1:]

        return dev_review, dev_attr_label, dev_senti_label

    def load_test_data(self):

        assert os.path.exists(self.configs['testa_data_path']) and os.path.getsize(self.configs['testa_data_path']) > 0

        with open(self.configs['testa_data_path'], 'rb') as f:
            testa_review = pickle.load(f)

        return testa_review

    def dim(self):
        print(self.attribute_dic)
        print('shape of train attr label: ',np.shape(self.train_attr_label))
        print(self.train_attr_label[0])


if __name__ == "__main__":
    dal = DimAttrLabel()
    dal.dim()