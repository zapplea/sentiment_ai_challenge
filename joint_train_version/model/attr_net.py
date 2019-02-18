import tensorflow as tf
import model.layer as layers
import util.operation as op

class Attr_Net(object):

    def __init__(self, config):
        self.config = config

    def build(self, review, rev_len, sent_len, attr_label, is_training=False):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        sent_emb = layers.sent_sru(self.config, review, sent_len, name='attr')

        # score:(batch, attr_num, 2)  attr_not_mention:(batch, attr_num, emb_dim)
        # att:(batch, attr, rev_len)
        self.score, self.doc_att = self.attr_esim_match(sent_emb, rev_len, 'attr_doc')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score,labels=tf.cast(attr_label,dtype=tf.int32)))
        self.attr_pred = tf.cast(tf.argmax(self.score, axis=-1), dtype=tf.float32)

        return

    def attr_esim_match(self, rev, rev_len, name):

        doc_emb = layers.doc_sru(self.config, rev, rev_len, 'attr')

        score, att  = self.ESIM(doc_emb, rev_len)

        return score, att

    def ESIM(self, R_s, R_len, mask_value=-2 ** 32 + 1):
        """
        
        :param R_s: (batch, rev, emb)
        :param R_len: (batch)
        :param mask_value:
        :return:
        """

        C_s = tf.get_variable(
            name='attr_context',
            shape=[self.config['attribute_num'], self.config['rnn_dim']],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        )
        not_mention = tf.get_variable(
            name='not_mention',
            shape=[self.config['attribute_num'], self.config['rnn_dim']],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        )

        # attr_context:(batch,attr,emb)
        C_s = tf.tile(tf.expand_dims(C_s, axis=0), multiples=[self.config['batch_size'], 1, 1])
        # attr_context_len:(batch)
        C_len = tf.ones(shape=[self.config['batch_size']]) * self.config['attribute_num']
        # not mention:(batch, attr,emb)
        not_mention = tf.tile(tf.expand_dims(not_mention, axis=0), multiples=[self.config['batch_size'], 1, 1])

        att = tf.einsum('bak,bik->bai',C_s,R_s)
        mask = op.mask(C_len,R_len,C_s.shape[-2],R_s.shape[-2])
        # att:(batch, attr, rev_len)
        att = tf.nn.softmax(att * mask + (1.0-mask) * mask_value, axis=-1)

        C_d = tf.einsum('bai,bik->bak',att,R_s)

        attr_feature = tf.concat([C_s, C_d, C_s-C_d, C_s*C_d], axis=-1)
        not_mention_feature = tf.concat([not_mention, C_d, not_mention-C_d, not_mention*C_d], axis=-1)

        attr_score = []
        not_mention_score = []
        for i in range(self.config['attribute_num']):
            attr_score.append(tf.layers.dense(
                inputs=attr_feature[:, i],
                units=1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                  uniform=True),
                bias_initializer=tf.zeros_initializer,
                name='pred_' + str(i),
                reuse=tf.AUTO_REUSE
            ))
            not_mention_score.append(tf.layers.dense(
                not_mention_feature[:, i],
                units=1,
                name='pred_',
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                  uniform=True),
                bias_initializer=tf.zeros_initializer,
                reuse=tf.AUTO_REUSE
            ))
        attr_score = tf.stack(attr_score, axis=1)
        not_mention_score = tf.stack(not_mention_score, axis=1)

        score = tf.concat([not_mention_score, attr_score], axis=-1)


        return score, att

