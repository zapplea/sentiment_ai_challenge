import pickle
import os

class SubDataset:
    def __init__(self,):
        self.up = 500
        self.data_config = {'train_data_file_path':
                                '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl',
                            'out_train_data_file_path':
                                '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar_trail.pkl',
                            'test_data_file_path':
                                '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_dev_data_withChar.pkl',
                            'out_test_data_file_path':
                                '/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_dev_data_withChar_trail.pkl'}

    def load_train_data(self):
        with open(self.data_config['train_data_file_path'], 'rb') as f:
            review, char_review, attr_label, senti_label, attribute_dic, word_dic, table, char_table = pickle.load(f)
        with open(self.data_config['out_train_data_file_path'],'wb') as f:
            pickle.dump((review[:self.up], char_review[:self.up], attr_label[:self.up], senti_label[:self.up], attribute_dic, word_dic, table, char_table),f)


    def load_test_data(self):
        with open(self.data_config['test_data_file_path'], 'rb') as f:
            review, char_review, attr_label, senti_label = pickle.load(f)

        with open(self.data_config['out_test_data_file_path'],'wb') as f:
            pickle.dump((review[:self.up], char_review[:self.up], attr_label[:self.up], senti_label[:self.up]),f)

    def main(self):
        self.load_train_data()
        self.load_test_data()


if __name__ == "__main__":
    subD = SubDataset()
    subD.main()