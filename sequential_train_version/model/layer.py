import tensorflow as tf
import util.operation as op

def attr_net_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['max_sent_len']],
        name='review'
    )

    attr_label = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['attribute_num']],
        name='attr_label'
    )

    is_training = tf.placeholder(
        tf.bool,
        shape=[],
        name='is_training'
    )

    table = tf.placeholder(
        tf.float32,
        shape=[_conf['vocab_size'], _conf['emb_dim']],
    )

    return review, attr_label, is_training, table

def senti_net_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.int32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['max_sent_len']],
        name='review'
    )

    sent_att = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['max_rev_len']],
        name='sent_att'
    )

    attr_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num']],
        name='attr_label'
    )

    senti_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['sentiment_num']],
        name='senti_label'
    )

    is_training = tf.placeholder(
        tf.bool,
        shape=[])

    table = tf.placeholder(
        tf.float32,
        shape=[_conf['vocab_size'], _conf['emb_dim']],
    )

    return review, sent_att, attr_label, senti_label, is_training, table

def senti_net_bert_input(_conf):
    # define placehloders
    review = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['max_rev_len'], _conf['bert_dim']])

    sent_att = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['max_rev_len']])

    attr_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num']])

    senti_label = tf.placeholder(
        tf.float32,
        shape=[_conf['batch_size'], _conf['attribute_num'], _conf['sentiment_num']])

    is_training = tf.placeholder(
        tf.bool,
        shape=[])

    return review, sent_att, attr_label, senti_label, is_training

def sent_lstm(config, review_embed, sent_mask, name):

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(review_embed, shape=[-1, config['max_sent_len'], config['emb_dim']])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_mask, shape=[-1, ])

    with tf.variable_scope(name+'_sent', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )
        bw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=sent_emb,
            sequence_length=sent_len,
            dtype=tf.float32)

        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.concat([tf.reduce_mean(outputs, axis=1), tf.reduce_max(outputs, axis=1)], axis=-1)

        outputs = tf.layers.dense(
            outputs,
            units=config['emb_dim'],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                              uniform=True),
            bias_initializer=tf.zeros_initializer,
            name='dense',
            reuse=tf.AUTO_REUSE
        )

        # outputs:(batch, rev_len, emb_dim)
        outputs = tf.reshape(outputs, shape=[-1, config['max_rev_len'], config['emb_dim']])

    return outputs

def sent_sru(config, review_embed, sent_mask, name):

    # sent_emb:(batch*rev,sent,emb)
    sent_emb = tf.reshape(review_embed, shape=[-1, config['max_sent_len'], config['emb_dim']])
    # sent_emb:(batch*rev,)
    sent_len = tf.reshape(sent_mask, shape=[-1, ])

    with tf.variable_scope(name+'_sent', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )
        bw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=sent_emb,
            sequence_length=sent_len,
            dtype=tf.float32)

        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.concat([tf.reduce_mean(outputs, axis=1), tf.reduce_max(outputs, axis=1)], axis=-1)

        outputs = tf.layers.dense(
            outputs,
            units=config['emb_dim'],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                              uniform=True),
            bias_initializer=tf.zeros_initializer,
            name='dense',
            reuse=tf.AUTO_REUSE
        )

        # outputs:(batch, rev_len, emb_dim)
        outputs = tf.reshape(outputs, shape=[-1, config['max_rev_len'], config['emb_dim']])

    return outputs

def doc_lstm(config, rev, rev_len, name):

    with tf.variable_scope(name+'_doc', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )
        bw_cell = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'] / 2,
            initializer=tf.orthogonal_initializer,
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=rev,
            sequence_length=rev_len,
            dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)

    return outputs

def doc_sru(config, rev, rev_len, name):

    with tf.variable_scope(name+'_doc', reuse=tf.AUTO_REUSE):
        # define parameters
        fw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )
        bw_cell = tf.contrib.rnn.SRUCell(
            config['rnn_dim'] / 2
        )

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=rev,
            sequence_length=rev_len,
            dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)

    return outputs


def attention(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K) / tf.sqrt(1546.0)  # [batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K) / tf.sqrt(1546.0)

    if is_mask:
        mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
        logits = mask * logits + (1 - mask) * mask_value

    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        print('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)


def FFN(x, out_dimension_0=None, out_dimension_1=None):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    with tf.variable_scope('FFN_1', reuse=tf.AUTO_REUSE):
        y = op.dense(x, out_dimension_0)
        y = tf.nn.relu(y)
    with tf.variable_scope('FFN_2', reuse=tf.AUTO_REUSE):
        z = op.dense(y, out_dimension_1)  # , add_bias=False)  #!!!!
    return z


def block(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_layer_norm=True,
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''

    # att.shape = (batch_size, max_turn_len, emb_size)
    att = attention(Q, K, V,
                    Q_lengths, K_lengths,
                    attention_type='dot',
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm', reuse=tf.AUTO_REUSE):
            y = op.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y)
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm', reuse=tf.AUTO_REUSE):
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, max_turn_len, emb_size)
    return w