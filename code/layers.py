#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras, layers, models

class Average(tf.Module):
	def __init__(self, word_emb):
		super(Average, self).__init__()
		self.word_emb = word_emb

	def forward(self, input):
		"""
		input: (data, lengths): (IntTensor(batch_size, max_sent_len), IntTensor(batch_size))
		"""
		data, lengths = input
		embeds = self.word_emb(data)
