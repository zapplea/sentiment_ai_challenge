import tensorflow as tf
import model.layer as layers
import util.operation as op
from model.AOA import AOA

class Senti_Net(object):

    def __init__(self, config):
        self.config = config
        self.AOA = AOA(self.config)

    def build(self, review, rev_len, sent_len, sent_att, attr_label, senti_label, is_training=False):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        # senti_doc_emb:(batch, attr, emb_dim)
        senti_doc_emb = self.AOA.generate_final_rep(review,sent_len,sent_att)

        self.senti_score = []
        for i in range(self.config['attribute_num']):
            self.senti_score.append(tf.squeeze(tf.layers.dense(
                inputs=senti_doc_emb[:, i],
                units=3,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                  uniform=True),
                bias_initializer=tf.zeros_initializer,
                name='pred_senti_' + str(i),
                reuse=tf.AUTO_REUSE
            )))
        self.senti_score = tf.stack(self.senti_score, axis=1)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.senti_score, labels=senti_label, name='senti_loss')
        self.loss = tf.reduce_mean(self.loss * attr_label)

        # pred:(batch, attr_num)
        self.senti_pred = tf.argmax(self.senti_score, axis=-1)

        return

