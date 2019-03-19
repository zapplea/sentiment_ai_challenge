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
    parser.add_argument('--char_mod',default='cwe',choices=['cwe','cwep'],type=str)
    args = parser.parse_args()
    lr = [1e-3,1e-4,1e-5]
    reg = [1e-3,1e-4,1e-5,1e-6]
    char_vocab_size = {'cwe':9565,'cwep':28693}
    data_file_path = {'cwe':{'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/cwe_merged_train.pkl',
                             'test_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/cwe_merged_dev.pkl'},
                      'cwep':{'train_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/cwep_merged_train.pkl',
                             'test_data_file_path':'/datastore/liu121/sentidata2/data/aic2018_junyu/cwep_merged_dev.pkl'}}
    print('char mod:%s\nchar vocab size:%d'%(args.char_mod,char_vocab_size[args.char_mod]))
    config = {'model':{'biSRU':{'shared_layers_num':2,
                                'separated_layers_num':3,
                                'rnn_dim':500,
                                'char_rnn_dim':200},
                       'lr': lr[args.lr],
                       'reg_rate':reg[args.reg],
                       "vocab_size": 400631,
                       'word_dim':300,
                       'max_sent_len':1141,
                       'attr_num':20,
                       'senti_num':4,
                       'padding_word_index':0,
                       'clip_value':10.0,
                       'max_word_len':11,
                       'char_vocab_size':char_vocab_size[args.char_mod],
                       'char_dim':200,
                       'padding_char_index':0},
              'train':{'epoch_num':100,
                       'report_rootPath':'/datastore/liu121/sentidata2/report/aic_junyu',
                       'report_fname':'%sbaseWithMaxPoolingChar_report_reg%s_lr%s.info'%(args.char_mod,str(reg[args.reg]),str(lr[args.lr])),
                       'sr_rootPath': '/datastore/liu121/sentidata2/result/aic_junyu',
                       'sr_fname':'%sbaseWithMaxPoolingChar_ckpt_reg%s_lr%s/model.ckpt'%(args.char_mod,str(reg[args.reg]),str(lr[args.lr])),
                       'early_stop_limit':5,
                       'mod':1,
                       'attributes_num':20,},
              'datafeeder':{'batch_size':10,
                            'train_data_file_path':data_file_path[args.char_mod]['train_data_file_path'],
                            'test_data_file_path':data_file_path[args.char_mod]['test_data_file_path']}}
    main(config)