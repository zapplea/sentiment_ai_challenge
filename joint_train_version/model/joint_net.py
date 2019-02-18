import tensorflow as tf
import model.layer as layers
import util.operation as op
from model.attr_net import Attr_Net
from model.senti_net import Senti_Net

class Joint_Net(object):

    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config
        self.A_Net = Attr_Net(self.config)
        self.S_Net = Senti_Net(self.config)

    def build_graph(self):

        if self.config['rand_seed'] is not None:
            rand_seed = self.config['rand_seed']
            tf.set_random_seed(rand_seed)
            print('set tf random seed: %s' % self.config['rand_seed'])

        with self.graph.as_default():

            self.review, self.attr_label, self.senti_label, self.is_training, self.table = layers.joint_net_input(self.config)

            # convert table to variable
            table_v = tf.Variable(self.table, name='table')

            # review_embed : (batch, rev_len, sent_len, emb_dim)
            review_embed = tf.nn.embedding_lookup(table_v, self.review)

            # rev_len:(batch,)
            # sent_len:(batch, rev)
            rev_len, sent_len = op.generate_mask(self.config, self.review)

            with tf.variable_scope('attr_net',reuse=tf.AUTO_REUSE):
                self.A_Net.build(review_embed, rev_len, sent_len, self.attr_label)

            with tf.variable_scope('senti_net',reuse=tf.AUTO_REUSE):
                self.S_Net.build(review_embed, rev_len, sent_len, self.A_Net.doc_att, self.attr_label, self.senti_label)

            self.joint_loss = self.A_Net.loss + self.S_Net.loss

            # Calculate cross-entropy loss
            tv = tf.trainable_variables()
            for v in tv:
                print(v)

            self.global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = self.config['learning_rate']
            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step=self.global_step,
                decay_steps=300,
                decay_rate=0.9,
                staircase=True)

            Optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = Optimizer.minimize(
                self.joint_loss,
                global_step=self.global_step)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
            self.all_variables = tf.global_variables()
            self.grads_and_vars = Optimizer.compute_gradients(self.joint_loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    print(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
            self.g_updates = Optimizer.apply_gradients(
                self.capped_gvs,
                global_step=self.global_step)

        return self.graph