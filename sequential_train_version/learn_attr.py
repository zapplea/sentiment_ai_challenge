import getpass
import sys
import os
# if getpass.getuser() == 'yibing':
#     sys.path.append('/home/yibing/Documents/code/nlp/sentiment_ai_challenge')
# elif getpass.getuser() == 'lujunyu':
#     sys.path.append('/home/lujunyu/repository/sentiment_coarse_model')
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# elif getpass.getuser() == 'liu121':
#     sys.path.append('/home/liu121/sentiment_ai_challenge')

from model.attr_net import Attr_Net
import bin.attr_train as attr_train

# configure
model_data_path = '/hdd/lujunyu/dataset/meituan/'
model_path = '/hdd/lujunyu/model/meituan/'

conf = {
    'train_data_path' : os.path.join(model_data_path, 'train_han_fasttext.pkl'),
    'dev_data_path' : os.path.join(model_data_path, 'dev_han_fasttext.pkl'),
    'testa_data_path' : os.path.join(model_data_path, 'testa_han_fasttext.pkl'),

    "init_model": None, #should be set for test

    "rand_seed": None,
    "learning_rate":1e-3,
    "attribute_threshold":0.5,
    "vocab_size": 266078,    #111695
    "emb_dim": 300,
    "batch_size": 50, #200 for test

    "max_rev_len": 25,
    "max_sent_len": 251,
    'attribute_num': 20,
    'sentiment_num': 3,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 0, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising

    "rnn_dim":300,

    'multi_head':8,

    'Model': 'D_HAN_MC'
}
conf.update({'save_path' : os.path.join(model_path, conf['Model'] + '/attr/')})


model = Attr_Net(conf)
attr_train.train(conf, model)


