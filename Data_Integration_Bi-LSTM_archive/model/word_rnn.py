import tensorflow as tf
from tensorflow.contrib import rnn


class WordRNN(object):
    def __init__(self, max_document_length, num_class,vocab_size,embedding_size,trainset_embedding=None):
        self.lr = 0.001
        # self.embedding_size = 256
        self.num_hidden = 512
        self.fc_num_hidden = 256

        self.x = tf.placeholder(tf.int32, [None, max_document_length],name = 'input_x')
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.float32, [None,num_class], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, [],name = 'dropout')
        self.x_init = tf.placeholder(tf.float32, shape=(vocab_size,embedding_size),name = 'x_init')
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")




        with  tf.device("/cpu:0"),tf.variable_scope("embedding"):

            embeddings = tf.Variable(self.x_init, dtype=tf.float32,trainable=True,name="pretrained_embedding")
            train_embeddings = tf.Variable(trainset_embedding, dtype=tf.float32,
                                           trainable=True,name="embs_only_in_train")

            embeddings = tf.concat([embeddings, train_embeddings], axis=0)
            x_emb = tf.nn.embedding_lookup(embeddings, self.x)
            print('....',x_emb)

        with tf.variable_scope("rnn"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)  # backward direction cell
            # cell = rnn.BasicLSTMCell(self.num_hidden)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,lstm_bw_cell, x_emb, sequence_length=self.x_len, dtype=tf.float32)
            # rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            #     lstm_fw_cell, lstm_bw_cell, x_emb, sequence_length=self.x_len, dtype=tf.float32)

            # Concat output
            lstm_concat = tf.concat(rnn_outputs, axis=2)  # [batch_size, sequence_length, lstm_hidden_size * 2]
            lstm_out = tf.reduce_mean(lstm_concat, axis=1)  # [batch_size, lstm_hidden_size * 2]


            # cell = rnn.BasicLSTMCell(self.num_hidden)
            # rnn_outputs, _ = tf.nn.dynamic_rnn(
            #     cell, x_emb, sequence_length=self.x_len, dtype=tf.float32)
            # rnn_output_flat = tf.reshape(rnn_outputs, [-1, max_document_length * self.num_hidden])

        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(lstm_out, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)

        # with tf.name_scope("output"):
        #     self.logits = tf.layers.dense(dropout, num_class)
        #     self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
        #     self.scores = tf.sigmoid(self.logits, name="scores")

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[self.fc_num_hidden, num_class],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_class], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(dropout, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")
            # self.predictions = tf.argmax(self.logits, -1, output_type=tf.float32)

        with tf.name_scope("loss"):

            self.loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y),axis = 1))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

        # with tf.name_scope("accuracy"):
        #
        #     correct_predictions = tf.equal(self.predictions, self.y)
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("lr"):
            self.lr = tf.Variable(0.001, name='learning_rate', trainable=False)
