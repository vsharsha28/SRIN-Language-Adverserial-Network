#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np
import os

from options import opt

class Vocab:
	"""
attributes: (self)
		vocab_size
		emb_size
		embeddings 
		w2vvocab = {word : idx}
		v2wvocab = [idx => emb_vector]
		pt_w2vvocab = {word : idx}
		pt_v2wvocab = [idx => emb_vector]
		unk_tok
		unk_idx
		eos_tok
		eos_idx

methods: (self)
		__init__(self, pre_train_infile)
		base_form(word)
		new_rand_emb(self)
		init_embed_layer(self)
		add_word(self, word)
		clear_pretrained_vectors(self)
		lookup(self, word)
		get_word(self, i)

	"""

	def __init__(self, pre_train_infile):
		"""
		load pre-trained words - embedding vectors in pt_***vocabs and initialise ***vocabs

		"""
		self.vocab_size = 0
		self.emb_size = opt.emb_size
		self.embeddings = []
		self.w2vvocab = {}
		self.v2wvocab = []
		
		# load pretrained embedings
		if(os.path.isfile(pre_train_infile)):
			with open(pre_train_infile, 'r') as infile:

				first_line = infile.readline().split()
				assert len(first_line) == 2
				n_vecs, emb_dim = map(int, first_line)	# first line has total number of vectors and embedding dimensions
				assert emb_dim == self.emb_size

				# add an UNK token
				self.pretrained = np.empty((n_vecs, emb_dim), dtype=np.float)
				self.pt_v2wvocab = []
				self.pt_w2vvocab = {}
				cnt = 0
				for line in infile:
					parts = line.rstrip().split(' ')
					word = parts[0]
					#if word in self.pt_v2wvocab: continue		# no need to check
					# add to vocabs
					self.pt_v2wvocab.append(word)
					self.pt_w2vvocab[word] = cnt
					vector = [float(x) for x in parts[1:]]
					self.pretrained[cnt] = vector
					cnt += 1
		else:
			pass
		
		# add <unk>
		self.unk_tok = '<unk>'
		self.add_word(self.unk_tok)
		self.unk_idx = self.w2vvocab[self.unk_tok]
		# add EOS token
		self.eos_tok = '</s>'
		self.add_word(self.eos_tok)
		opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
		self.embeddings[self.eos_idx][:] = 0

	def base_form(word):
		"""
		return stripped and lowercased word
		"""
		return word.strip().lower()

	def new_rand_emb(self):
		"""
		return a normal random emb_vector
		"""
		vec = np.random.normal(-1, 1, size=self.emb_size)
		vec /= sum(x*x for x in vec) ** .5
		return vec

	def add_word(self, word):
		"""
		add new word to the ***vocab. \nUse this to only add new words or used words from pt_***vocabs to ***vocabs.
		"""
		word = Vocab.base_form(word)
		if word not in self.w2vvocab:
			if not opt.random_emb and hasattr(self, 'pt_w2vvocab'):
				if opt.fix_unk and word not in self.pt_w2vvocab:
					# use fixed unk token, do not update vocab
					return
				if word in self.pt_w2vvocab:
					vector = self.pretrained[self.pt_w2vvocab[word]].copy()
				else:
					vector = self.new_rand_emb()
			else:
				vector = self.new_rand_emb()
			self.v2wvocab.append(word)
			self.w2vvocab[word] = self.vocab_size
			self.embeddings.append(vector)
			self.vocab_size += 1

	def clear_pretrained_vectors(self):
		"""
		clear the pretrained vectors and pt_***vocab
		"""
		del self.pretrained
		del self.pt_w2vvocab
		del self.pt_v2wvocab

	def lookup(self, word):
		"""
		return value of word (word_idx) from w2vvocab
		"""
		word = Vocab.base_form(word)
		if word in self.w2vvocab:
			return self.w2vvocab[word]
		return self.unk_idx

	def get_word(self, i):
		"""
		return emb_vector at index i
		"""
		return self.v2wvocab[i]

	def init_embed_layer(self):
		"""
		clear pretrained vectors and return an embedding layer initialized with self.embeddings
		"""
		# free some memory
		#self.clear_pretrained_vectors()
		emb_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_size, name='vocab_embedding')
		emb_layer.build(input_shape=(self.vocab_size, self.emb_size))
		emb_layer.set_weights(tf.convert_to_tensor(self.embeddings, dtype=tf.float32))
		emb_layer.trainable = False
		#assert len(emb_layer.weight) == self.vocab_size
		return emb_layer


if __name__ == "__main__":
	vocab = Vocab('')
	vocab.add_word('hey')
	vocab.add_word('hi')
	emb_layer = vocab.init_embed_layer()
	print(emb_layer.variables, '\n', emb_layer.weights)
