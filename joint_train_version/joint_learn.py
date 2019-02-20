import getpass
import sys
import os

from model.joint_net import Joint_Net
import bin.joint_train as joint_train

# configure
model_data_path = '/datastore/liu121/sentidata2/data/aic2018_junyu'
model_path = '/datastore/liu121/sentidata2/result/aic_junyu'

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
    "batch_size": 20, #200 for test

    "max_rev_len": 25,
    "max_sent_len": 251,
    'attribute_num': 20,
    'sentiment_num': 3,
    'attribute_prototype_num':3,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 0, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising

    "rnn_dim":300,

    'multi_head':8,

    'Model': 'D_HAN_MC'
}
conf.update({'save_path' : os.path.join(model_path, conf['Model'] + '/joint/')})


model = Joint_Net(conf)
joint_train.train(conf, model)


