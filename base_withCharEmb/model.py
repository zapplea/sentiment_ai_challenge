import tensorflow as tf
from layers import Layers

class Model:
    def __init__(self,config):
        self.config = config
        self.layers = Layers(config)
    def build_senti_net(self):
        X_id = self.layers.X_input()
        char_X_id = self.layers.char_X_input()
        senti_Y = self.layers.senti_Y_input()

        table = self.layers.word_embedding_table()
        mask = self.layers.padded_word_mask(X_id)
        # (batch size, max sent len, word dim)
        X = self.layers.lookup(X_id,table,mask)
        seq_len = self.layers.sequence_length(X_id)

        char_table = self.layers.char_embedding_table()
        char_mask = self.layers.padded_char_mask(char_X_id)
        # (batch size, max sent len, max word len, char dim)
        char_X = self.layers.lookup(char_X_id,char_table,char_mask)
        print('char id: ',char_X_id.get_shape())
        char_seq_len = self.layers.char_sequence_length(char_X_id)
        print('char seq len: ',char_seq_len.get_shape())
        exit()

        # char_biSRU
        bisru_name = 'char_biSRU'
        # (batch size, max sent len, max word len, char dim)
        char_X = self.layers.biSRU(char_X, char_seq_len, dim = self.config['model']['biSRU']['char_rnn_dim'], name=bisru_name)
        # (batch size, max sent len, char dim)
        char_X = self.layers.max_pooling(char_X,char_mask)
        # (batch size, max sent len, word dim + char dim)
        X = tf.concat([X,char_X],axis=-1)
        bisru_name = 'share'
        for i in range(self.config['model']['biSRU']['shared_layers_num']):
            # (batch size, max sent len, rnn_dim)
            X = self.layers.biSRU(X,seq_len,dim=self.config['model']['biSRU']['rnn_dim'],name=bisru_name)
        graph = tf.get_default_graph()
        tf.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
            graph.get_tensor_by_name('biSRU_%s/bidirectional_rnn/fw/sru_cell/kernel:0'%bisru_name)))
        tf.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
            graph.get_tensor_by_name('biSRU_%s/bidirectional_rnn/bw/sru_cell/kernel:0'%bisru_name)))
        # (batch size, rnn dim)
        sent_repr_ls = []
        for i in range(self.config['model']['biSRU']['separated_layers_num']):
            bisru_name = 'sep_layer'+str(i)
            X = self.layers.biSRU(X,seq_len, dim=self.config['model']['biSRU']['rnn_dim'], name=bisru_name)
            tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
                graph.get_tensor_by_name('biSRU_%s/bidirectional_rnn/fw/sru_cell/kernel:0' % bisru_name)))
            tf.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(
                graph.get_tensor_by_name('biSRU_%s/bidirectional_rnn/bw/sru_cell/kernel:0' % bisru_name)))
            # (attr num, rnn dim)
            A = self.layers.attr_matrix(name=bisru_name)
            # (batch size, attr num, max sent len)
            att = self.layers.attention(A,X, X_id)
            # (batch size, attr num, rnn dim)
            sent_repr = self.layers.sent_repr(att, X)
            sent_repr_ls.append(sent_repr)
        # (batch size, attr num, sep bisru layers num * rnn dim)
        sent_repr = tf.concat(sent_repr_ls,axis=2)
        senti_score = self.layers.senti_score(sent_repr)
        pred = self.layers.senti_prediction(senti_score)
        loss = self.layers.senti_loss(senti_score,senti_Y)
        train_step = tf.train.AdamOptimizer(self.config['model']['lr']).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        return {'loss':loss,'pred':pred,'graph':tf.get_default_graph(),'train_step':train_step,'saver':saver}