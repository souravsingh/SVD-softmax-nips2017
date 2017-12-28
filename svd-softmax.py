import tensorflow as tf
import numpy as np

class SVD-Softmax(object):
    
    def __init__(self, tgt_vocab_size, hidden_units, window_size=2 ** 5, num_full_view=2 ** 11):
        """
        initialize SVD
        :param tgt_vocab_size: int, num of vocabulary
        :param hidden_units: int, num of hidden units
        :param window_size: int, width of preview window W( hidden_units/ 8 is recommended)
        :param num_full_view: int, num of full-view size
        :return: A Tensor [batch_size, seq_length, tgt_vocab_size], output after softmax approximation
        """

        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_units = hidden_units
        self.window_size = window_size
        self.num_full_view = num_full_view

        self.b = tf.Variable(
            tf.truncated_normal([self.tgt_vocab_size, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="b_SVD", trainable=False)
        self.V_t = tf.Variable(
            tf.truncated_normal([self.hidden_units, self.hidden_units],
                                stddev=1.0 / math.sqrt(hidden_units)), name="V_SVD", trainable=False)
                                
        
    def update_params(self, weights):
        """
        update svd parameter b, V_t
        :param weights: output weight of softmax
        """
        _s, U, V = tf.svd(weights, full_matrices=False)
        self.b.assign(tf.matmul(U, tf.diag(_s)))
        self.V_t.assign(tf.transpose(V))
