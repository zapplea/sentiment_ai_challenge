import tensorflow as tf
from layers import Layers

class Model:
    def __init__(self,config):
        self.config = config
        self.layers = Layers(config)
    def build_senti_net(self):
        X_id = self.layers.X_input()
        senti_Y = self.layers.senti_Y_input()
        table = self.layers.word_embedding_table()
        mask = self.layers.padded_word_mask(X_id)
        X = self.layers.lookup(X_id,table,mask)
        seq_len = self.layers.sequence_length(X_id)
        for i in range(self.config['model']['biSRU']['layers_num']):
            # (batch size, max sent len, rnn_dim)
            X = self.layers.biSRU(X,seq_len)
        graph = tf.get_default_graph()
        tf.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
            graph.get_tensor_by_name('biSRU/bidirectional_rnn/fw/sru_cell/kernel:0')))
        tf.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
            graph.get_tensor_by_name('biSRU/bidirectional_rnn/bw/sru_cell/kernel:0')))
        # (batch size, rnn dim)
        A = self.layers.attr_matrix()
        att = self.layers.attention(A,X)
        sent_repr = self.layers.sent_repr(att, X)
        senti_score = self.layers.senti_score(sent_repr)
        pred = self.layers.senti_prediction(senti_score)
        loss = self.layers.senti_loss(senti_score,senti_Y)
        train_step = tf.train.AdamOptimizer(self.config['lr']).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        return {'loss':loss,'pred':pred,'graph':tf.get_default_graph(),'train_step':train_step,'saver':saver}