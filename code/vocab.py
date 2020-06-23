#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np
import os, io
from pathlib import Path
os.chdir(os.path.dirname(__file__))

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

	def __init__(self, pre_train_infile = None):
		"""
		load pre-trained words - embedding vectors in pt_***vocabs and initialise ***vocabs

		"""
		self.vocab_size = 0
		self.emb_size = opt.emb_size
		self.embeddings = []
		self.w2vvocab = {}
		self.v2wvocab = []
		
		# load pretrained embedings
		if(pre_train_infile is not None):				#if(os.path.isfile(pre_train_infile)):
			with io.open(Path(pre_train_infile), 'r', encoding='utf-8') as infile:

				first_line = infile.readline().split()
				assert len(first_line) == 2
				n_vecs, emb_dim = map(int, first_line)	# first line has total number of vectors and embedding dimensions
				#assert emb_dim == self.emb_size
				self.emb_size = emb_dim
				# add an UNK token
				self.pretrained = np.empty((n_vecs, emb_dim), dtype=np.float)
				self.pt_v2wvocab = []
				self.pt_w2vvocab = {}
				cnt = 0
				#lno = 1
				while True:
					#lno += 1
					#print(lno)
					line = infile.readline()
					if not line: break
					parts = line.rstrip().split(' ')
					word = parts[0]
					#if word in self.pt_v2wvocab: continue		# no need to check if assumed no repetition mistake
					# add to vocabs
					self.pt_v2wvocab.append(word)
					self.pt_w2vvocab[word] = cnt
					vector = [float(x) for x in parts[1:]]
					self.pretrained[cnt] = vector
					cnt += 1
		else:
			print("pre_train_file ", Path(pre_train_infile), " does not exist.")
		
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

	def add_word(self, word, vec = None):
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
					vector = self.new_rand_emb() if vec is None else vec
			else:
				vector = self.new_rand_emb() if vec is None else vec
			self.v2wvocab.append(word)
			self.w2vvocab[word] = self.vocab_size
			self.embeddings.append(vector)
			self.vocab_size += 1

	def add_text(line):
		"""
		add new text (words) to the vocabularies
		"""
		for word in line.strip.split():
			self.add_word(word)

	def text_to_sequence():
		pass

	def clear_pretrained_vectors(self):
		"""
		clear the pretrained vectors and pt_***vocab
		"""
		if hasattr(self, 'pretrained'): del self.pretrained
		if hasattr(self, 'pt_w2vvocab'): del self.pt_w2vvocab
		if hasattr(self, 'pt_v2wvocab'): del self.pt_v2wvocab

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
		self.clear_pretrained_vectors()
		emb_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_size, input_length=opt.max_seq_len, name='vocab_embedding')
		emb_layer.build(input_shape=(None, self.vocab_size, self.emb_size))
		emb_layer.set_weights(np.array([self.embeddings], dtype=float))
		emb_layer.trainable = False
		assert emb_layer.weights[0].shape[0] == self.vocab_size, "layer weights len not equal to vocab size in layer " + emb_layer.name
		return emb_layer


if __name__ == "__main__":
	"""
	run as main
	"""
	vocab = Vocab(Path(opt.pre_trained_emb_file))
	vocab.add_word('the')
	vocab.add_word('of')
	vocab.add_word('this')
	emb_layer = vocab.init_embed_layer()
	print(emb_layer.variables)