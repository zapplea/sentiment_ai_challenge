import pickle
import os
import numpy as np

class DimAttrLabel:
    def __init__(self):
        self.configs={'train_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/train_han_fasttext.pkl',
                      'testa_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/testa_han_fasttext.pkl',
                      'dev_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/dev_han_fasttext.pkl',
                      'new_train_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/new_train_data.pkl',
                      'new_dev_data_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/new_dev_data.pkl'
                      }
        self.train_review, self.train_attr_label, self.train_senti_label, self.attribute_dic, self.word_dic, self.table = self.load_train_data()
        self.dev_review, self.dev_attr_label, self.dev_senti_label = self.load_dev_data()

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

    def process_attr_labels(self,id_to_attribute,new_attribute_to_id,attr_label):
        shape = np.shape(attr_label)
        new_attr_label = []
        for i in range(shape[0]):
            new_label_instance = np.zeros(shape=(len(new_attribute_to_id),),dtype='float32')
            label_instance = attr_label[i]
            for j in range(shape[1]):
                key = new_attribute_to_id[id_to_attribute[j]]
                if label_instance[j] == 1:
                    new_label_instance[key] = label_instance[j]
            new_attr_label.append(new_label_instance)
        return np.array(new_attr_label).astype('float32')

    def dim(self):
        id_to_attribute = {}
        for key in self.attribute_dic:
            attr = key.split('_')[0]
            id_to_attribute[self.attribute_dic[key]] = attr

        new_attribute_to_id = {}
        count = 0
        for key in self.attribute_dic:
            attr = key.split('_')[0]
            if attr not in new_attribute_to_id:
                new_attribute_to_id[attr]=count
                count+=1
        train_attr_label = self.process_attr_labels(id_to_attribute,new_attribute_to_id,self.train_attr_label)
        dev_attr_label = self.process_attr_labels(id_to_attribute, new_attribute_to_id, self.dev_attr_label)
        with open(self.configs['new_train_data_path'],'wb') as f:
            for data in [self.attribute_dic,]:
                pickle.dump(data,f)
        with open(self.configs['new_dev_data_path'],'wb') as f:
            for data in [self.dev_review, dev_attr_label, self.dev_attr_label, self.dev_senti_label]:
                pickle.dump(data,f)

if __name__ == "__main__":
    dal = DimAttrLabel()
    dal.dim()