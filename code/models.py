#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import os, io
from pathlib import Path
os.chdir(os.path.dirname(__file__))

from options import opt
from layers import *


class DAN_Fetaure_Extractor(keras.Model):
	def __init__(self, vocab, num_layers, hidden_size, pooling='avg', dropout=0, batch_norm=False):
		super(DAN_Feature_Extractor, self).__init__()

		if(pooling == 'sum' or pooling == 'add'): self.pool = Summing(self.emb_layer)
		else: self.pool = Averaging(self.emb_layer)

		assert num_layers >= 0, 'Invalid layer numbers'

		self.fcnet = keras.Sequential()
		for i in range(num_layers):
			if dropout > 0:
				self.fcnet.add(layers.Dropout(rate=dropout))

			if i == 0:
				self.fcnet.add(layers.Dense(units=hidden_size, input_shape=(vocab.emb_size,), activation='selu'))
			else:
				self.fcnet.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation='selu'))

			if batch_norm:
				self.fcnet.add(layers.BatchNormalization(input_shape=(hidden_size,)))	# same shape as input	# use training=False when making inference from model (model.predict, model.evaluate?)
			
			self.fcnet.add(layers.LeakyReLU(alpha=0.3))
			#self.fcnet.add(layers.ReLU())
	
	def call(self, input):
		return self.fcnet(self.pool(input))



class LSTM_Feature_Extractor(keras.Model):
	def __init__(self, vocab, num_layers, hidden_size, dropout=0, bidir_rnn=True, attn_type='dot'):
		super(LSTM_Feature_Extractor, self).__init__()

		self.num_layers = num_layers
		self.bidir_rnn = bidir_rnn
		self.attn_type = attn_type
		self.hidden_size = hidden_size//2 if bdrnn else hidden_size
		self.n_cells = self.num_layers*2 if bdrnn else self.num_layers

		self.emb_layer = vocab.init_embed_layer()

		if bidir_rnn:
			self.rnn = layers.Bidirectional(layers.LSTM(units=self.hidden_size, num_layers=num_layers, dropout=dropout, input_shape=(vocab.emb_size,)))
		else:
			self.rnn = layers.LSTM(units=self.hidden_size, num_layers=num_layers, dropout=dropout, input_shape=(vocab.emb_size,))

		if attn_type == 'dot':
			self.attn = layers.Attention()
		elif attn_type == 'add':
			self.attn = layers.AdditiveAttention()

	def call(self, inputs):
		data, lengths = inputs
		lengths_list = lengths.tolist()
		batch_size = len(data)
		emb_layer = self.emb_layer(data)
		packed = pack_padded_sequence(emb_layer, lengths_list, batch_first=True)
		state_shape = self.n_cells, batch_size, self.hidden_size
		h0 = c0 = embeds.data.new(*state_shape)
		output, (ht, ct) = self.rnn(packed, (h0, c0))

		if self.attn_type == 'last':
			return ht[-1] if not self.bdrnn \
						else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
		elif self.attn_type == 'avg':
			unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
			return torch.sum(unpacked_output, 1) / lengths.float().view(-1, 1)
		elif self.attn_type == 'dot':
			unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
			return self.attn((unpacked_output, lengths))
		else:
			raise Exception('Please specify valid attention (pooling) mechanism')



class CNN_Feature_Extractor(keras.Model):
	def __init__(self, vocab, num_layers, hidden_size, kernel_num, kernel_sizes, dropout):
		super(CNNFeatureExtractor, self).__init__()
		self.emb_layer = vocab.init_embed_layer()
		self.kernel_num = kernel_num
		self.kernel_sizes = kernel_sizes

		self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
		
		assert num_layers >= 0, 'Invalid layer numbers'
		self.fcnet = nn.Sequential()
		for i in range(num_layers):
			if dropout > 0:
				self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
			if i == 0:
				self.fcnet.add_module('f-linear-{}'.format(i),
						nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
			else:
				self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
			self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

	def call(self, inputs):
		data, lengths = inputs
		batch_size = len(data)
		embeddings = self.emb_layer(data)
		# conv
		embeddings = tf.expand_dims(embeddings, axis=1) # batch_size, 1, seq_len, emb_size
		x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
		x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)
		# fcnet
		return self.fcnet(x)




class Sentiment_Classifier(keras.Model):
	def __init__(self, num_layers, hidden_size, output_size, dropout=0, batch_norm=False):
		super(Sentiment_Classifier, self).__init__()
		assert num_layers >= 0, 'Invalid layer numbers'

		self.net = models.Sequential()

		for _ in range(num_layers):
			if dropout > 0:
				self.net.add(layers.Dropout(rate=dropout))

			self.net.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation='selu'))

			if batch_norm:
				self.net.add(layers.BatchNormalization())

			self.net.add(layers.ReLU())

		self.net.add(layers.Dense(hidden_size, output_size))

		self.net.add(LogSoftmax(axis=-1))

	def call(self, input):
		return self.net(input)




class Language_Detector(keras.Model):
	def __init__(self, num_layers, hidden_size, dropout=0, batch_norm=False):
		super(LanguageDetector, self).__init__()
		assert num_layers >= 0, 'Invalid layer numbers'
		
		self.net = nn.Sequential()

		for i in range(num_layers):
			if dropout > 0:
				self.net.add(layers.Dropout(rate=dropout))
			
			self.net.add(layers.Dense(units=hidden_size, input_shape=(hidden_size,), activation='selu'))

			if batch_norm:
				self.net.add(layers.BatchNormalization(input_shape=(hidden_size,)))

		self.net.add(layers.Dense(units=output_size, input_shape=(hidden_size,), activation='selu'))

		self.net.add(layers.Dense(units=1, input_shape=(hidden_size,), activation='selu'))

	def call(self, input):
		return self.net(input)