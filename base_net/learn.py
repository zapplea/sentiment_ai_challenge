from model import Model
from senti_datafeeder import DataFeeder
from train import SentiTrain

def main(config):
    model_dic = Model.build_senti_net(config)
    datafeeder = DataFeeder(config['datafeeder'])
    train = SentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    data_root = ''
    save_root = ''
    config = {'model':{'biSRU':{'layers_num':2,
                                'rnn_dim':300},
                       'lr': 1e-3,
                       'reg_rate':1e-4,
                       "vocab_size": 266078,
                       'word_dim':300,
                       'max_sent_len':1141,
                       'attr_num':20,
                       'senti_num':3,
                       'padding_word_index':0,},
              'train':{'epoch_num':20,
                       'report_filePath':'/datastore/liu121/sentidata2/report/aic_junyu',
                       'early_stop_limit':1,
                       'mod':1,
                       'sr_path':'/datastore/liu121/sentidata2/result/aic_junyu',
                       'attributes_num':20,},
              'datafeeder':{'batch_size':50,
                            'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_train_data_trail.pkl',
                            'test_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_dev_data_trail.pkl'}}