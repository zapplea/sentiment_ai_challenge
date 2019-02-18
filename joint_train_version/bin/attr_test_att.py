import os
import time

import tensorflow as tf
import numpy as np
import json

import util.evaluation as eva
from util.data_generator import DataGenerator
import util.operation as op


def test(conf, _model):
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # config display
    print('configurations: %s' % conf)

    # Data Generate
    dg = DataGenerator(conf)
    print('Train data size: ', dg.train_data_size)
    print('Dev data size: ', dg.dev_data_size)
    print('Test data size: ', dg.test_data_size)

    # refine conf
    train_batch_num = int(dg.train_data_size / conf["batch_size"])
    val_batch_num = int(dg.dev_data_size / conf["batch_size"])

    conf["train_steps"] = conf["num_scan_data"] * train_batch_num
    conf["save_step"] = int(max(1., train_batch_num / 10))
    conf["print_step"] = int(max(1., train_batch_num / 50))

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build graph')
    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build graph sucess')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=_graph, config=config) as sess:
        sess.run(_model.init, feed_dict={_model.table: dg.table})
        if conf["init_model"]:
            model_path = tf.train.latest_checkpoint(conf["init_model"])
            _model.saver.restore(sess, model_path)
            print("sucess init attr model %s" % model_path)

        dev_score_file_path = conf['save_path'] + 'test_score'
        print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        dev_score_file = open(dev_score_file_path, 'w')
        all_attr_pred = []
        dev_loss = 0
        visualize_sample = []
        # caculate dev score
        for batch_index in range(val_batch_num):
            dev_review, dev_attr_label, dev_senti_label = dg.dev_data_generator(batch_index)
            feed = {
                _model.review: dev_review,
                _model.attr_label: dev_attr_label,
                _model.is_training:False
            }

            attr_loss, attr_scores, attr_pred ,doc_att = sess.run(
                [_model.loss, _model.score, _model.attr_pred, _model.for_senti_att],
                feed_dict=feed
            )
            dev_loss +=attr_loss

            visualize_sample.append(op.doc_att_visualization(conf, doc_att, dev_review, dev_attr_label, attr_pred, dg, batch_index))

            all_attr_pred.append(attr_pred)
        dev_score_file.close()
        with open(conf['save_path'] + 'visualization.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(visualize_sample, indent=4, ensure_ascii=False))
        all_attr_pred = np.concatenate(all_attr_pred,axis=0)
        cm = op.attr_confusion_matrix(all_attr_pred, dg.dev_attr_label, dg)
        # write evaluation result
        attr_f1, attr_class_f1 = eva.attr_evaluate(all_attr_pred, dg.dev_attr_label)
        dev_result_file_path = conf["save_path"] + "test_result"
        result = {'attr F1 score': attr_f1}
        for i, att in enumerate(dg.attribute_dic):
            assert dg.attribute_dic[att] == i
            result[att] = attr_class_f1[i]
        with open(dev_result_file_path, 'w') as f:
            f.write(json.dumps(result, indent=4))

        print('finish dev evaluation')
        print('val loss: ', dev_loss/val_batch_num)

        print('attr result: ', attr_f1)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

