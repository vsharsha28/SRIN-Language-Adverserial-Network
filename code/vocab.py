# vocab.py

#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing

import numpy as np
import os, io, sys, logging
from pathlib import Path
from tqdm import tqdm, trange
os.chdir(os.path.dirname(__file__))

from options import *

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

	def __init__(self, pre_train_infile = None, vecs = opt.n_vecs):
		"""
		load pre-trained words - embedding vectors in pt_***vocabs and initialise ***vocabs

		"""
		self.vocab_size = 0
		self.emb_size = opt.emb_size
		self.embeddings = []
		self.w2vvocab = {}
		self.v2wvocab = []
		
		self.pt_v2wvocab = []
		self.pt_w2vvocab = {}
		self.cnt = 0

		## load pretrained embedings
		#if(pre_train_infile is not None):				#if(os.path.isfile(pre_train_infile)):
		#	log.info('reading pre-trained embeddings from ' + pre_train_infile + '...')
		#	with io.open(Path(pre_train_infile), 'r', encoding='utf-8') as infile:

		#		first_line = infile.readline().split()
		#		assert len(first_line) == 2
		#		n_vecs, emb_dim = map(int, first_line)	# first line has total number of vectors and embedding dimensions
		#		assert emb_dim == self.emb_size
		#		self.emb_size = emb_dim
		#		if vecs is not None and vecs > 0: n_vecs = min(n_vecs, vecs)
		#		for _ in trange(n_vecs):
		#			line = infile.readline()
		#			if not line: break
		#			parts = line.rstrip().split(' ')
		#			word = parts[0]
		#			#if word in self.pt_v2wvocab: continue		# no need to check if assumed no repetition mistake
		#			# add to vocabs
		#			self.pt_v2wvocab.append(word)
		#			self.pt_w2vvocab[word] = cnt
		#			vector = [float(x) for x in parts[1:]]
		#			self.pretrained[cnt] = vector
		#			cnt += 1
		#	log.info("vectors imported...")
		#else:
		#	log.info("pre_train_file ", Path(pre_train_infile), " does not exist.\nSkipping...")
		
		# add <unk>
		self.unk_tok = opt.unk_tok
		self.add_word(self.unk_tok)
		opt.unk_idx = self.unk_idx = self.w2vvocab[self.unk_tok]
		self.embeddings[self.unk_idx][:] = 0
		# add BOS token
		self.bos_tok = opt.bos_tok
		self.add_word(self.bos_tok)
		opt.bos_idx = self.bos_idx = self.w2vvocab[self.bos_tok]
		self.embeddings[self.bos_idx][:] = 0
		# add EOS token
		self.eos_tok = opt.eos_tok
		self.add_word(self.eos_tok)
		opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
		self.embeddings[self.eos_idx][:] = 1	# 0
		log.info("vocab initializing...done.")
		# add pre trained embeddings
		self.add_pre_trained_emb(pre_train_infile, vecs)

	def add_pre_trained_emb(self, pre_train_infile = None, vecs = opt.n_vecs):
		# load pretrained embedings
		if(pre_train_infile is None): raise Exception('file not specified...')
		if(os.path.isfile(pre_train_infile)):
			log.info('reading pre-trained embeddings from ' + pre_train_infile + '...')
			with io.open(Path(pre_train_infile), 'r', encoding='utf-8') as infile:
				first_line = infile.readline().split()
				assert len(first_line) == 2
				n_vecs, emb_dim = map(int, first_line)	# first line has total number of vectors and embedding dimensions
				assert emb_dim == self.emb_size
				self.emb_size = emb_dim
				if vecs is not None and vecs > 0: n_vecs = min(n_vecs, vecs)
				if not hasattr(self, 'pretrained'):	self.pretrained = np.empty(shape=(n_vecs, emb_dim), dtype=np.float)
				else: self.pretrained = np.append(self.pretrained, np.empty(shape=(n_vecs, emb_dim), dtype=np.float), axis=0)
				for _ in trange(n_vecs):
					line = infile.readline()
					if not line: break
					parts = line.rstrip().split(' ')
					word = parts[0]
					#if word in self.pt_v2wvocab: continue		# no need to check if assumed no repetition mistake
					# add to vocabs
					self.pt_v2wvocab.append(word)
					self.pt_w2vvocab[word] = self.cnt
					vector = [float(x) for x in parts[1:]]
					self.pretrained[self.cnt] = vector
					self.cnt += 1
			log.info("embedding vectors imported...")
		else:
			raise FileNotFoundError(log.info("pre_train_file ", Path(pre_train_infile), " does not exist..."))

	def base_form(self, word):
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
		word = self.base_form(word=word)
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
		return self.w2vvocab[word]


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
	

	def hash_fit_on_text(self, line):
		"""
		fit on text (sentence) with tf.keras.preprocessing.text.hashing_trick\n(NOT TESTED)
		"""
		return preprocessing.text.hashing_trick(line, n=self.vocab_size, hash_function=self.lookup, filters='')


	def text_to_sequence(self, line, update_vocab=True):
		"""
		convert text (line) to sequence
		"""
		return [self.add_word(w) for w in line.strip().split()] if update_vocab else\
			[self.lookup(w) for w in line.strip().split()]


	def text_list_to_sequence(self, text_list, update_vocab=True):
		"""
		convert text (word list) to sequence
		"""
		return [self.add_word(w) for w in text_list] if update_vocab else\
			[self.lookup(w) for w in text_list]


	def fit_on_text(self, line_list):	# tf.keras.preprocessing.text.hashing_trick
		"""
		add new text (sentence) to the vocabularies
		"""
		return [[self.add_word(w) for w in line.strip().split()] for line in line_list]


	def fit_on_text_list(self, texts_list):	# tf.keras.preprocessing.text.hashing_trick
		"""
		add new text (sentence) to the vocabularies
		"""
		return [[self.add_word(w) for w in line] for line in texts_list]


	def pad_text_list(self, text_list, max_len=opt.max_seq_len, pad='pre', truncate='post', add_eos_tok=False):
		"""
		pad single text (words) list
		"""
		if add_eos_tok: max_len -= 1
		text_list = text_list[:max_len]
		text_list = ['<s>' for _ in range(max_len - len(text_list))] + text_list
		if add_eos_tok: text_list += ['</s>']
		return text_list


	def pad_sequences(self, dataset, max_len=opt.max_seq_len, pad='pre', truncate='post', add_eos=False):
		"""
		pad list of sequences (sequence : list of int) with keras.preprocessing.sequence.pad_sequences
		"""
		if add_eos: max_len -= 1
		seq_list, stars = zip(*dataset)
		seq_list = self.fit_on_text_list(seq_list)
		seq_list = preprocessing.sequence.pad_sequences(tqdm(seq_list), maxlen=max_len, truncating='post', value=opt.bos_idx)
		if add_eos: seq_list = preprocessing.sequence.pad_sequences(tqdm(seq_list), maxlen=max_len+1, padding='post', value=opt.eos_idx)
		return tf.data.Dataset.from_tensor_slices((np.array(seq_list), np.array(stars)))

	def clear_pretrained_vectors(self):
		"""
		clear the pretrained vectors and pt_***vocab
		"""
		if hasattr(self, 'pretrained'): del self.pretrained
		if hasattr(self, 'pt_w2vvocab'): del self.pt_w2vvocab
		if hasattr(self, 'pt_v2wvocab'): del self.pt_v2wvocab
	

	def init_embed_layer(self, clear_pt=True):
		"""
		clear pretrained vectors and return an embedding layer initialized with self.embeddings
		"""
		if clear_pt: self.clear_pretrained_vectors()
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
	print(opt.pre_trained_src_emb_file)
	vocab = Vocab(opt.pre_trained_src_emb_file)
	vocab.add_pre_trained_emb(opt.pre_trained_tgt_emb_file)
	vocab.add_word('the')
	vocab.add_word('of')
	vocab.add_word('this')
	emb_layer = vocab.init_embed_layer(clear_pt=False)
	print(emb_layer.variables)

# imp link https://www.tensorflow.org/tfx/tutorials/transform/census , https://www.tensorflow.org/api_docs/python/tf/numpy_function , https://www.tensorflow.org/api_docs/python/tf/py_function