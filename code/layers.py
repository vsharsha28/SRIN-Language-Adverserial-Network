#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np


@keras_export('keras.layers.Averaging')
class Averaging(layers.Layer):
	def __init__(self, emb_layer):
		super(Average, self).__init__()
		self.emb_layer = emb_layer

	def call(self, inputs):
		"""
		inputs => input_seq_batch : Tensor(batch_size, max_sent_len), seq_lengths : Tensor(batch_size)
		"""
		input_seq_batch, seq_lengths = inputs
		embeddings = self.emb_layer(input_seq_batch)
		lengths = tf.resape(lengths, (-1, 1)).broadcast_to(X.shape)
		self.W = tf.cast(tf.reduce_sum(embeddings, axis=1), dtype=tf.float32)
		lengths = tf.broadcast_to(tf.reshape(lengths, (-1, 1)), shape=self.W.shape)
		return self.W/tf.cast(lengths, dtype=tf.float32)


@keras_export('keras.layers.Summing')
class Summing(layers.Layer):
	def __init__(self, emb_layer):
		super(Summing, self).__init__()
		self.emb_layer = emb_layer

	def call(self, inputs):
		"""
		inputs => (data, lengths): (Tensor(batch_size, max_sent_len), Tensor(batch_size))
		"""
		data, _ = inputs
		embeddings = self.emb_layer(data)
		self.W = tf.cast(tf.reduce_sum(embeddings, axis=1), dtype=tf.float32)
		return self.W


@keras_export('keras.layers.DotAttention')
class DotAttention(layers.Layer):
	def __init__(self, hidden_size):
		super(DotAttentionLayer, self).__init__()
		self.hidden_size = hidden_size
		self.W = layers.Dense(units=1, input_shape=(hidden_size,), use_bias=False)

	def call(self, inputs):
		"""
		inputs => unpacked_padded_output: np.array(batch_size, seq_len, hidden_size), lengths: np.array(batch_size)
		"""
		data, lengths = inputs
		batch_size, max_len, _ = data.size()
		flat_input = data.contiguous().view(-1, self.hidden_size)
		logits = tf.reshape(self.W(flat_input), shape=(batch_size, max_len))
		alphas = tf.nn.softmax(logits, dim=-1)

		# computing mask
		idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).to(inputs.device)
		idxes = tf.expand_dims(tf.cast(tf.linspace(start=0, stop=max_len), dtype=tf.int64), axis=0)
		mask = tf.cast((idxes < tf.expand_dims(lengths, axis=1)), dtype=tf.float32)

		alphas = alphas * mask
		# renormalize
		alphas = tf.reshape(alphas / tf.reduce_sum(alphas, axis=1, keepdims=True), shape=(-1, 1))
		output = tf.squeeze(tf.einsum('bnm,bmp->bnp', tf.expand_dims(alphas, axis=1), data), axis=1)
		#output = tf.squeeze(tf.matmul(tf.expand_dims(alphas, axis=1), data), axis=1)		# batch matrix multiplication
		return output



@keras_export('keras.layers.LogSoftmax')
class LogSoftmax(layers.Layer):
	"""LogSoftmax activation function.
	Input shape:
		Arbitrary. Use the keyword argument `input_shape`
			(tuple of integers, does not include the samples axis)
			when using this layer as the first layer in a model.
	Output shape:
		Same shape as the input.
	Arguments:
		axis: Integer, axis along which the softmax normalization is applied.
	"""

	def __init__(self, axis=-1, **kwargs):
		super(LogSoftmax, self).__init__(**kwargs)
		self.supports_masking = True
		self.axis = axis

	def call(self, inputs):
		return tf.nn.log_softmax(inputs, axis=self.axis)

	def get_config(self):
		config = {'axis': self.axis}
		base_config = super(LogSoftmax, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@tf_utils.shape_type_conversion
	def compute_output_shape(self, input_shape):
		return input_shape