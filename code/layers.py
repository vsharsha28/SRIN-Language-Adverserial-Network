#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras, layers, models

class Average(keras.Model):
	def __init__(self, emb_layer):
		super(Average, self).__init__()
		self.emb_layer = emb_layer

	def call(self, input_seq_batch : tf.Tensor):
		"""
		input: (data, lengths): (IntTensor(batch_size, max_sent_len), IntTensor(batch_size))
		"""
		#embs = self.emb_layer(input) #, length
		#X = tf.reduce_sum(embs, axis=1)
		#lengths (pass as parameter) = tf.resape(lengths, (-1, 1)).broadcast_to(X.shape)
		embeddings = self.emb_layer(input_seq_batch)
		self.W = tf.reduce_mean(tf.cast(embeddings, dtype=tf.float32), axis=1)
		return self.W

class Summing(layers.Layer):




class DotAttention(layers.Layer):
