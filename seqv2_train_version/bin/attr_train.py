import os
import time

import tensorflow as tf
import numpy as np
import json

import util.evaluation as eva
from util.data_generator2 import DataGenerator
import util.operation as op


def train(conf, _model):
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
            _model.saver.restore(sess, conf["init_model"])
            print("sucess init %s" % conf["init_model"])

        average_loss = 0.0
        step = 0
        best_result = 0

        for step_i in range(conf["num_scan_data"]):
            for batch_index in range(train_batch_num):
                train_review, train_attr_label, train_senti_label = dg.train_data_generator(batch_index)
                feed = {
                    _model.review: train_review,
                    _model.attr_label: train_attr_label,
                    _model.is_training: True
                }
                batch_index = (batch_index + 1) % train_batch_num

                _, curr_loss, curr_lr  = sess.run([_model.g_updates, _model.loss, _model.learning_rate], feed_dict=feed)

                average_loss += curr_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                          " processed: [" + str(step * 1.0 / train_batch_num) +
                          "] loss: [" + str(average_loss / conf["print_step"]) + "]" +
                          "] lr: [" + str(curr_lr) + "]")
                    average_loss = 0
                    # op.attr_att_visualization(conf, curr_attr_att, train_review, train_label, curr_pred, dg)

                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())), '  Save step: %s' % index)

                    all_attr_pred = []
                    dev_loss = 0
                    # caculate dev score
                    for batch_index in range(val_batch_num):
                        dev_review, dev_attr_label, dev_senti_label = dg.dev_data_generator(batch_index)
                        feed = {
                            _model.review: dev_review,
                            _model.attr_label: dev_attr_label,
                            _model.is_training:False
                        }

                        attr_loss, attr_pred  = sess.run([_model.loss, _model.attr_pred],feed_dict=feed)
                        dev_loss +=attr_loss

                        all_attr_pred.append(attr_pred)
                    all_attr_pred = np.concatenate(all_attr_pred,axis=0)
                    # write evaluation result
                    attr_f1, attr_class_f1 = eva.attr_evaluate(all_attr_pred, dg.dev_attr_label)
                    dev_result_file_path = conf["save_path"] + "result." + str(index)
                    result = {'attr F1 score': attr_f1}
                    for i, att in enumerate(dg.attribute_dic):
                        assert dg.attribute_dic[att] == i
                        result[att] = attr_class_f1[i]
                    with open(dev_result_file_path, 'w') as f:
                        f.write(json.dumps(result, indent=4))

                    print('finish dev evaluation')
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print('val loss: ', dev_loss/val_batch_num)

                    if attr_f1 > best_result:
                        best_result = attr_f1
                        print('best attr result: ', attr_f1)
                        _save_path = _model.saver.save(sess, conf["save_path"] + "model.ckpt." + str(step / conf["save_step"]))
                        print("succ saving model in " + _save_path)
                        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

