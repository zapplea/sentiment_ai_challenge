# -*- coding: utf-8 -*-

import tensorflow as tf
import model.layer as layers

class AOA(object):

    def __init__(self, config):
        self.conf = config
        self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',uniform=True)

    def generate_final_rep(self, review_embed, sent_len, sent_att):
        """
        
        :param review_embed: 
        :param sent_len: (batch, rev)
        :param sent_att: (batch, attr, rev)
        :return: 
        """

        #
        sent = self.Bi_LSTM(review_embed, sent_len)

        #
        attr_c = self.init_attribute()

        # sent_rep:(batch, attr, rev, emb)
        sent_rep = self.interaction(sent, sent_len, attr_c)
        sent_rep = tf.reshape(sent_rep,shape=[-1,self.conf['max_rev_len'],self.conf['attribute_num'],self.conf['rnn_dim']])
        sent_rep = tf.transpose(sent_rep, perm=[0,2,1,3])

        # f_rep:(batch, attr, emb)
        # use attr net attention weight
        f_rep = tf.einsum('bark,bar->bak',sent_rep,sent_att)

        return f_rep


    def Bi_LSTM(self, review_embed, sent_len):
        # sent_emb:(batch*rev,sent,emb)
        sent_emb = tf.reshape(review_embed, shape=[-1, self.conf['max_sent_len'], self.conf['emb_dim']])
        # sent_emb:(batch*rev,)
        sent_len = tf.reshape(sent_len, shape=[-1, ])

        # define parameters
        fw_cell = tf.contrib.rnn.SRUCell(
            self.conf['rnn_dim'] / 2
        )
        bw_cell = tf.contrib.rnn.SRUCell(
            self.conf['rnn_dim'] / 2
        )

        sent_emb, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=sent_emb,
            sequence_length=sent_len,
            dtype=tf.float32)

        # sent_emb:(batch*rev,sent,emb)
        sent_emb = tf.concat(sent_emb, axis=-1)

        return sent_emb

    def init_attribute(self):

        attr_c = tf.get_variable(
            name='context',
            shape=[self.conf['attribute_num'], self.conf['attribute_prototype_num'], self.conf['emb_dim']],
            dtype=tf.float32,
            initializer=self.initializer
        )

        return attr_c

    def interaction(self, sent, sent_len, attr_c, mask_value=-2 ** 32 + 1):
        """
        
        :param sent: (batch*rev, sent, emb)
        :param sent_len: (batch, rev)
        :param attr_c: (attr, attr_proto, emb)
        :return: 
        """

        # att:(batch*rev, attr, sent, attr_proto)
        att = tf.einsum('bnk,amk->banm',sent,attr_c)
        # mask
        # mask_1:(batch*rev, sent)
        mask_1 = tf.sequence_mask(tf.reshape(sent_len, shape=[-1]),maxlen=self.conf['max_sent_len'],dtype=tf.float32)
        mask_2 = tf.ones_like(attr_c[:,:,0],dtype=tf.float32)
        mask = tf.einsum('bn,am->banm',mask_1,mask_2)
        att = att * mask + mask_value * (1-mask)


        alpha = tf.nn.softmax(att, axis=-2)
        beta = tf.nn.softmax(att, axis=-1)

        _beta = tf.reduce_mean(beta, axis=-2)

        gamma = tf.einsum('banm,bam->ban',alpha,_beta)

        r = tf.einsum('bnk,ban->bak',sent, gamma)

        return r

