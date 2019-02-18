import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from model.senti_net import Senti_Net
from model.attr_net import Attr_Net
import bin.senti_test as sent_test

# configure
model_data_path = '/hdd/lujunyu/dataset/meituan/'
model_path = '/hdd/lujunyu/model/meituan/'

conf = {
    'train_data_path' : os.path.join(model_data_path, 'train_han_fasttext.pkl'),
    'dev_data_path' : os.path.join(model_data_path, 'dev_han_fasttext.pkl'),
    'testa_data_path' : os.path.join(model_data_path, 'testa_han_fasttext.pkl'),

    "attr_init_model": '/hdd/lujunyu/model/meituan/D_HAN_MC/attr/',
    "senti_init_model": '/hdd/lujunyu/model/meituan/D_HAN_MC/senti/', #should be set for test

    "rand_seed": 1,
    "learning_rate":1e-3,
    "attribute_threshold":0.5,
    "vocab_size": 266078,    #111695
    "emb_dim": 300,
    "batch_size": 1, #200 for test

    "max_rev_len": 25,
    "max_sent_len": 251,
    'attribute_num': 20,
    'sentiment_num': 3,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 0, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising

    "rnn_dim":300,
    'bert_dim':768*2,

    'multi_head':8,

    'Model': 'D_HAN_MC'
}
conf.update({'save_path' : os.path.join(model_path, conf['Model'] + '/senti/')})

attr_model = Attr_Net(conf)
senti_model = Senti_Net(conf)
sent_test.test(conf, attr_model, senti_model)


