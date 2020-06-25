# layers.py

#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np

from options import *
from vocab import *
from data import *
from utils import *

class Averaging(layers.Layer):
	def __init__(self, emb_layer, **kwargs):
		super(Averaging, self).__init__(**kwargs)
		self.emb_layer = emb_layer

	def call(self, input_seq_batch):
		"""
parameters:	input_seq_batch : Tensor(BSZ, MSL)
return:		Embeddings averaged over sequences (MSL axis) : Tensor(BSZ, EMBDIM)
		"""
		lengths = [len(x) for x in input_seq_batch]										# (BSZ,)
		embeddings = self.emb_layer(input_seq_batch)									# (BSZ, MSL, EMBDIM)
		lengths = tf.reshape(lengths, (-1, 1))											# (BSZ, 1)
		self.W = tf.cast(tf.reduce_sum(embeddings, axis=1), dtype=tf.float32)			# (BSZ, EMBDIM)
		#print(self.W.shape)
		lengths = tf.broadcast_to(tf.reshape(lengths, (-1, 1)), shape=self.W.shape)		# (BSZ, EMBDIM)
		return self.W/tf.cast(lengths, dtype=tf.float32)								# (BSZ, EMBDIM)


class Summing(layers.Layer):
	def __init__(self, emb_layer, **kwargs):
		super(Summing, self).__init__(**kwargs)
		self.emb_layer = emb_layer

	def call(self, inputs):
		"""
parameters:	input_seq_batch : Tensor(BSZ, MSL)
return:		Embeddings summed over sequences (MSL axis) : Tensor(BSZ, EMBDIM)
		"""
		embeddings = self.emb_layer(inputs)										# (BSZ, MSL, EMBDIM)
		self.W = tf.cast(tf.reduce_sum(embeddings, axis=1), dtype=tf.float32)	# (BSZ, EMBDIM)
		return self.W


class DotAttention(layers.Layer):
	def __init__(self, hidden_size, **kwargs):
		super(DotAttentionLayer, self).__init__(**kwargs)
		self.hidden_size = hidden_size
		self.dense = layers.Dense(units=1, input_shape=(hidden_size,), activation='linear', use_bias=False)	# input Tensor : (None, HIDSZ)

	def call(self, inputs):
		"""
		inputs => unpacked_padded_output: np.array(BSZ, MSL, HIDSZ)
		"""
		lengths = [len(x) for x in inputs]										# (BSZ,)
		BSZ, MSL, EMBDIM = inputs.shape
		#flat_input = data.contiguous().view(-1, self.hidden_size)
		try:
			flat_inputs = tf.convert_to_tensor([x for x in inputs.unbatch()])	# (BSZ*MSL, EMBDIM)
		except ValueError:
			flat_inputs = tf.convert_to_tensor([x for x in inputs])				# (1*MSL, EMBDIM)
		flat_input = tf.reshape(data, shape=(-1, self.hidden_size))				# (-1, HIDSZ)	:	(BSZ*MSL*EMBDIM/HIDSZ, HIDSZ)
		logits = tf.reshape(self.dense(flat_input), shape=(BSZ, MSL))			# (BSZ*MSL*EMBDIM, 1) => (BSZ, MSL)
		alphas = tf.nn.softmax(logits, dim=-1)

		# computing mask
		idxes = tf.expand_dims(tf.range(max_len, dtype=tf.int64), axis=0)
		mask = tf.cast((idxes < tf.expand_dims(lengths, axis=1)), dtype=tf.float32)

		alphas = alphas * mask
		# renormalize
		alphas = tf.reshape(alphas / tf.reduce_sum(alphas, axis=1, keepdims=True), shape=(-1, 1))
		output = tf.squeeze(tf.einsum('bnm,bmp->bnp', tf.expand_dims(alphas, axis=1), data), axis=1)
		#output = tf.squeeze(tf.matmul(tf.expand_dims(alphas, axis=1), data), axis=1)		# batch matrix multiplication
		return output



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

	def compute_output_shape(self, input_shape):
		return input_shape


if __name__ == "__main__":
	infile = Path('Amazon reviews/test/dataset_en_test.json')
	vocab = Vocab(opt.pre_trained_src_emb_file)
	rev = AmazonReviews()
	data = rev.load_data(lang='en', dat='train', lines=100)
	data = vocab.pad_sequences(data)
	print('\n', data)
	emb_layer = vocab.init_embed_layer()
	print(emb_layer(tf.convert_to_tensor([x for x, y in data.as_numpy_iterator()], dtype='int32')), '\n\n')
	avg = Averaging(emb_layer)
	print(avg(tf.convert_to_tensor([x for x, y in data.as_numpy_iterator()], dtype='int32')))