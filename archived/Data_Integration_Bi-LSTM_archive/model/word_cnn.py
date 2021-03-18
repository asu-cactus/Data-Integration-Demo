import tensorflow as tf


class WordCNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class):
        self.embedding_size = 128
        self.learning_rate = 0.001
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100
        self.fc_num_hidden = 256

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.float32, [None,num_class], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
            self.x_emb = tf.expand_dims(self.x_emb, -1)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.layers.conv2d(
                self.x_emb,
                filters=self.num_filters,
                kernel_size=[filter_size, self.embedding_size],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])


        with tf.name_scope("fc"):
            fc_output = tf.layers.dense(h_pool_flat, self.fc_num_hidden, activation=tf.nn.relu)
            dropout = tf.nn.dropout(fc_output, self.keep_prob)
        with tf.name_scope("output"):


            W = tf.Variable(tf.truncated_normal(shape=[self.fc_num_hidden, num_class],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_class], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(dropout, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")


        with tf.name_scope("loss"):


            self.loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y), axis=1))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

