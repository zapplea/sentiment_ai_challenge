import tensorflow as tf
import model.layer as layers
import util.operation as op

class Senti_Net(object):

    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config

    def build_graph(self):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        with self.graph.as_default():

            # review: (batch, rev_len, bert_dim)
            self.review, self.sent_att, self.attr_label, self.senti_label, self.is_training = layers.senti_net_bert_input(self.config)

            self.rev_len = op.generate_mask_for_bert(self.review)

            rev_emb = layers.block(self.review,self.review,self.review,self.rev_len,self.rev_len)

            # senti_doc_emb:(batch, attr, emb_dim)
            senti_doc_emb = self.doc_encoder(rev_emb, self.sent_att, 'senti')

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

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.senti_score, labels=self.senti_label, name='senti_loss')
            # mask loss
            self.loss = tf.reduce_mean(self.loss * self.attr_label)

            # pred:(batch, attr_num)
            self.senti_pred = tf.argmax(self.senti_score, axis=-1)

            # Calculate cross-entropy loss
            tv = tf.trainable_variables()
            for v in tv:
                print(v)

            self.global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = self.config['learning_rate']
            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step=self.global_step,
                decay_steps=200,
                decay_rate=0.8,
                staircase=True)

            Optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = Optimizer.minimize(
                self.loss,
                global_step=self.global_step)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
            self.all_variables = tf.global_variables()
            self.grads_and_vars = Optimizer.compute_gradients(self.loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    print(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
            self.g_updates = Optimizer.apply_gradients(
                self.capped_gvs,
                global_step=self.global_step)

        return self.graph


    def doc_encoder(self, rev, attr_att, name):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # rev_emb:(batch, attr, emb_dim)
            rev_emb = tf.einsum('bik,bai->bak', rev, attr_att)

        return rev_emb
