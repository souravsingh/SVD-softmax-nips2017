import tensorflow as tf
import numpy as np

class SVD-Softmax(object):
    
    def __init__(self, tgt_vocab_size, hidden_units, window_size=2 ** 5, num_full_view=2 ** 11):
        """
        Initialize SVD
        :param tgt_vocab_size: int, number of vocabulary
        :param hidden_units: int, number of hidden units
        :param window_size: int, width of preview window W( hidden_units/ 8 is recommended)
        :param num_full_view: int, num of full-view size
        :return: A Tensor output after softmax approximation
        """

        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_units = hidden_units
        self.window_size = window_size
        self.num_full_view = num_full_view

        self.b = tf.Variable(tf.truncated_normal([self.tgt_vocab_size, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="b_SVD", trainable=False)
        self.V_t = tf.Variable(tf.truncated_normal([self.hidden_units, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="V_SVD", trainable=False)
                                
        
    def update_params(self, weights):
        """
        update svd parameter b, V_t
        :param weights: output weight of softmax
        """
        _s, U, V = np.linalg.svd(weights, full_matrices=0)
        self.b.assign(tf.matmul(U, tf.diag(_s)))
        self.V_t.assign(tf.transpose(V))
    
    def get_output(self, dec_output, biases):
        """
        Obtains svd-softmax approximation
        :param dec: A Tensor [batch_size*seq_length, hidden_units], decoder output
        :param biases: A Tensor [tgt_vocab_size], output bias
        :return: A Tensor [batch_size*seq_length, tgt_vocab_size], output after softmax approximation
        """
        h = tf.einsum('ij,aj->ai', self.V_t, dec_output)
        z = tf.add(tf.einsum('ij,aj->ai', self.B[:, :self.window_size], _h[:, :self.window_size]), biases)

        top_k = tf.nn.top_k(z, k=self.tgt_vocab_size)
        indices, values = top_k.indices, top_k.values

        z = tf.add(tf.squeeze(tf.matmul(tf.gather(self.B, _indices[:, :self.num_full_view]), tf.expand_dims(_h, axis=-1))),
                    tf.gather(biases, _indices[:, :self.num_full_view]))
        z = tf.concat([_z, values[:, self.num_full_view:]], axis=-1)
        z = tf.map_fn(lambda x: tf.gather(x[0], tf.invert_permutation(x[1])), (_z, _indices), dtype=tf.float32)
        z = tf.exp(_z)
        Z = tf.expand_dims(tf.reduce_sum(_z, axis=-1), axis=1)
        logits = z / Z

        return logits
