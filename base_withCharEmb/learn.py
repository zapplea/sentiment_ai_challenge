from model import Model
from senti_datafeeder import DataFeeder
from train import SentiTrain
import tensorflow as tf
import argparse

def main(config):
    with tf.device('/gpu:0'):
        model = Model(config)
        model_dic = model.build_senti_net()
    datafeeder = DataFeeder(config['datafeeder'])
    train = SentiTrain(config,datafeeder)
    train.train(model_dic)

if __name__ == "__main__":
    # the label should be 4 dim
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=int)
    parser.add_argument('--reg',type=int)
    args = parser.parse_args()
    lr = [1e-3,1e-4,]
    reg = [1e-3,1e-4,1e-5,1e-6]
    config = {'model':{'biSRU':{'shared_layers_num':2,
                                'separated_layers_num':3,
                                'rnn_dim':200},
                       'lr': lr[args.lr],
                       'reg_rate':reg[args.reg],
                       "vocab_size": 266078,
                       'word_dim':200,
                       'max_sent_len':1141,
                       'attr_num':20,
                       'senti_num':4,
                       'padding_word_index':0,
                       'clip_value':10.0,
                       'max_word_len':11,
                       'char_vocab_size':10616,
                       'char_dim':50,
                       'padding_char_index':0},
              'train':{'epoch_num':100,
                       'report_filePath':'/datastore/liu121/sentidata2/report/aic_junyu',
                       'early_stop_limit':20,
                       'mod':1,
                       'sr_path':'/datastore/liu121/sentidata2/result/aic_junyu',
                       'attributes_num':20,},
              'datafeeder':{'batch_size':100,
                            'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_train_data_withChar.pkl',
                            'test_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/tenc_merged_dev_data_withChar.pkl'}}
    main(config)