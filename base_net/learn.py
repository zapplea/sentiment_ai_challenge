from model import Model
from senti_datafeeder import DataFeeder
from train import SentiTrain
import tensorflow as tf

def main(config):
    with tf.device('/gpu:0'):
        model = Model(config)
        model_dic = model.build_senti_net()
    datafeeder = DataFeeder(config['datafeeder'])
    train = SentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    config = {'model':{'biSRU':{'shared_layers_num':2,
                                'separated_layers_num':3,
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
                       'early_stop_limit':5,
                       'mod':1,
                       'sr_path':'/datastore/liu121/sentidata2/result/aic_junyu',
                       'attributes_num':20,},
              'datafeeder':{'batch_size':50,
                            'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_train_data.pkl',
                            'test_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/merged_dev_data.pkl'}}
    main(config)