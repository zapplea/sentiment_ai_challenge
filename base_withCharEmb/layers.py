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

    def char_X_input(self):
        X = tf.placeholder(shape=(None, self.config['model']['max_sent_len'],self.config['model']['max_word_len']),dtype='int32')
        tf.add_to_collection('senti_char_X_id',X)
        return X

    def padded_word_mask(self,X_id):
        """

        :param X_id: (batch size, max sent len)
        :return: (batch size, max sent len, word dim)
        """
        X_id = tf.cast(X_id, dtype='float32')
        padding_id = tf.ones_like(X_id, dtype='float32') * self.config['model']['padding_word_index']
        is_padding = tf.equal(X_id, padding_id)
        mask = tf.where(is_padding, tf.zeros_like(X_id, dtype='float32'), tf.ones_like(X_id, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self.config['model']['word_dim']])
        return mask

    def word_embedding_table(self):
        table = tf.placeholder(shape=(self.config['model']['vocab_size'], self.config['model']['word_dim']),dtype="float32")
        tf.add_to_collection('table', table)
        embedding = tf.Variable(table)
        return embedding

    def padded_char_mask(self, char_X_id):
        """

        :param char_X_id: (batch size, max sent len, max word len)
        :return:
        """
        char_X_id = tf.cast(char_X_id,dtype='float32')
        padding_id = tf.ones_like(char_X_id, dtype='float32') * self.config['model']['padding_char_index']
        is_padding = tf.equal(char_X_id, padding_id)
        mask = tf.where(is_padding, tf.zeros_like(char_X_id, dtype='float32'), tf.ones_like(char_X_id, dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask, axis=3), multiples=[1, 1, 1, self.config['model']['char_dim']])
        return mask


    def char_embedding_table(self):
        table = tf.placeholder(shape = (self.config['model']['char_vocab_size'],self.config['model']['char_dim']),dtype='float32')
        tf.add_to_collection('char_table',table)
        char_embedding = tf.Variable(table)
        return char_embedding

    def lookup(self,X_id,table,mask):
        """

        :param X_id: (batch size, max sent len) / (batch size, max sent len, max char len)
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

    def biSRU(self,X,seq_len,name=''):
        """

        :param X: (batch size, max sent len, word dim)
        :param seq_len: (batch size,)
        :param name:
        :return: (batch size, max sent len, rnn dim)
        """
        with tf.variable_scope('biSRU_'+name, reuse=tf.AUTO_REUSE):
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

    def attr_matrix(self,name=''):
        """

        :return: (attr num, rnn dim)
        """
        A = tf.get_variable(name='attr_matrix_'+name,
                            initializer=self.parameter_initializer(shape=(self.config['model']['attr_num'],self.config['model']['biSRU']['rnn_dim'])),
                            dtype='float32')
        # norm = tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(A)
        # tf.add_to_collection('reg', norm)
        return A

    def attention(self,A,X,X_id):
        """

        :param A: (attr num, rnn dim)
        :param X: (batch size, max sentence len, rnn dim)
        :param X_id: (batch size, max sentence len)
        :return: (batch size, attr num, max sent len)
        """
        X_id = tf.cast(X_id, dtype='float32')
        padding_id = tf.ones_like(X_id, dtype='float32') * self.config['model']['padding_word_index']
        is_padding = tf.equal(X_id, padding_id)
        # (batch size, max sentence len)
        mask = tf.where(is_padding,
                        tf.zeros_like(X_id, dtype='float32'),
                        tf.ones_like(X_id, dtype='float32'))
        # (batch size, max sentence len, attr num)
        temp = tf.clip_by_value(tf.tensordot(X,A,axes=[[2],[1]]),
                                clip_value_min=tf.constant(-self.config['model']['clip_value']),
                                clip_value_max=tf.constant(self.config['model']['clip_value']))
        # (attr num, batch size, max sent len)
        temp = tf.transpose(temp,perm=(2,0,1))
        temp = tf.multiply(mask,tf.exp(temp))
        # (attr num, batch size, 1)
        denominator = tf.reduce_sum(temp, axis=2, keepdims=True)
        # (attr num, batch size, max sent len)
        denominator = tf.tile(denominator,multiples=[1,1,self.config['model']['max_sent_len']])
        # (attr num, batch size, max sent len)
        att = tf.truediv(temp,denominator)
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

        :param sent_repr: (batch size, attr num, sep bisru layers num * rnn dim)
        :return: (batch size, attr num, senti num)
        """
        # (senti num, rnn dim)
        W = tf.get_variable(name='senti_score_W',initializer=self.parameter_initializer(shape=(self.config['model']['senti_num'],
                                                                                               self.config['model']['biSRU']['separated_layers_num']*
                                                                                               self.config['model']['biSRU']['rnn_dim']),
                                                                                        dtype='float32'))
        norm = tf.contrib.layers.l2_regularizer(self.config['model']['reg_rate'])(W)
        tf.add_to_collection('reg',norm)
        # (batch size, attr num, senti num)
        score = tf.tensordot(sent_repr,W,axes=[[2],[1]])

        return score

    def senti_prediction(self,score):
        """

        :param score: (batch size, attr num, senti num)
        :return:
        """
        # (batch size, attr num, senti num)
        temp = tf.nn.softmax(score,axis=-1)
        senti_pred = tf.where(tf.equal(tf.reduce_max(temp, axis=2, keep_dims=True), temp), tf.ones_like(temp),
                        tf.zeros_like(temp))
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