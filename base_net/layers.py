import tensorflow as tf

class Layers:
    def __init__(self,config):
        self.config = config

    def X_input(self):
        X = tf.placeholder(shape=(None,self.config['model']['max_sent_len']),dtype='int32')
        tf.add_to_collection('senti_X_id',X)
        return X

    def attr_Y_input(self):
        attr_Y = tf.placeholder(shape=(None,self.config['model']['attr_num']),dtype='int32')
        tf.add_to_collection('attr_Y',attr_Y)
        return attr_Y

    def senti_Y_input(self):
        senti_Y = tf.placeholder(shape=(None,self.config['model']['attr_num'],self.config['model']['senti_num']),dtype='int32')
        tf.add_to_collection('senti_Y',senti_Y)
        return senti_Y

    def padded_word_mask(self,X_id):
        """

        :param X_id: (batch size, max sent len)
        :return:
        """
        X_id = tf.cast(X_id, dtype='float32')
        padding_id = tf.ones_like(X_id, dtype='float32') * self.config['model']['padding_word_index']
        is_padding = tf.equal(X_id, padding_id)
        mask = tf.where(is_padding, tf.zeros_like(X_id, dtype='float32'), tf.ones_like(X_id, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.config['model']['word_dim']])
        return mask

    def word_embedding_table(self):
        table = tf.placeholder(shape=(self.config['model']['vocab_size'], self.config['model']['word_dim']))
        tf.add_to_collection('table', table)
        embedding = tf.Variable(table)
        return embedding

    def lookup(self,X_id,table,mask):
        """

        :param X_id: (batch size, max sent len)
        :param mask: used to prevent update of padded words
        :return:
        """
        X = tf.nn.embedding_lookup(table, X_id, partition_strategy='mod', name='lookup_table')
        X = tf.multiply(X,mask)
        return X

    def parameter_initializer(self,shape,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(shape[-1],dtype=dtype))
        init = tf.random_uniform(shape,minval=-stdv,maxval=stdv,dtype=dtype,seed=1)
        return init

    def sequence_length(self,X_id):
        """

        :param X_id: (batch size, max sentence len)
        :return:
        """
        padding_id = tf.ones_like(X_id, dtype='int32') * self.config['model']['padding_word_index']
        condition = tf.equal(padding_id, X_id)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(X_id, dtype='int32'), tf.ones_like(X_id, dtype='int32')),
                                axis=1, name='seq_len')
        return seq_len

    def biSRU(self,X,seq_len,):
        """

        :param X: (batch size, max sent len, word dim)
        :param seq_len: (batch size,)
        :param name:
        :return: (batch size, max sent len, rnn dim)
        """
        with tf.variable_scope('biSRU', reuse=tf.AUTO_REUSE):
            # define parameters
            fw_cell = tf.contrib.rnn.SRUCell(
                self.config['model']['biSRU']['rnn_dim'] / 2
            )
            bw_cell = tf.contrib.rnn.SRUCell(
                self.config['model']['biSRU']['rnn_dim'] / 2
            )

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=X,
                sequence_length=seq_len,
                dtype=tf.float32)

            outputs = tf.concat(outputs, axis=-1)
        return outputs

    def attr_matrix(self):
        """

        :return: (attr num, rnn dim)
        """
        A = tf.get_variable(name='attr_matrix',
                            initializer=self.parameter_initializer(shape=(self.config['model']['attr_num'],self.config['model']['biSRU']['rnn_dim'])),
                            dtype='float32')
        norm = tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(A)
        tf.add_to_collection('reg', norm)
        return A

    def attention(self,A,X):
        """

        :param A: (attr num, rnn dim)
        :param X: (batch size, max sentence len, rnn dim)
        :return:
        """
        # (batch size, max sentence len, attr num)
        temp = tf.matmul(X,A,transpose_b=True)
        # (attr num, batch size, max sent len)
        temp = tf.transpose(temp,perm=(2,0,1))
        # (attr num, batch size, max sent len)
        att = tf.nn.softmax(temp,axis=-1)
        # (batch size, attr num, max sent len)
        att = tf.transpose(att,perm=(1,0,2))
        return att

    def sent_repr(self,att,X):
        """

        :param att: (batch size, attr num, max sent len)
        :param X: (batch size, max sent len, rnn dim)
        :return:(batch size, attr num, rnn dim)
        """
        # (batch size, attr num, rnn dim)
        sent_repr = tf.matmul(att,X)
        return sent_repr

    def senti_score(self,sent_repr):
        """

        :param sent_repr: (batch size, attr num, rnn dim)
        :return: (batch size, attr num, senti num)
        """
        # (senti num, rnn dim)
        W = tf.get_variable(name='senti_score_W',initializer=self.parameter_initializer(shape=(self.config['model']['senti_num'],
                                                                                               self.config['model']['biSRU']['rnn_dim']),
                                                                                        dtype='float32'))
        norm = tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(W)
        tf.add_to_collection('reg',norm)
        # (batch size, attr num, senti num)
        score = tf.matmul(sent_repr,W,transpose_b=True)

        return score

    def senti_prediction(self,score):
        """

        :param score: ()
        :return:
        """
        # (batch size, attr num, senti num)
        senti_pred = tf.nn.softmax(score,axis=-1)
        return senti_pred

    def senti_loss(self,logits,labels):
        """

        :param logits: (batch size, attr num, senti num)
        :param labels: (batch size, attr num, senti num)
        :return:
        """
        reg = tf.get_collection('reg')
        loss = tf.reduce_mean(tf.add(
            tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=-1), axis=1),
            tf.reduce_sum(reg)))
        return loss