#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np

from options import opt

class Vocab:
	def __init__(self, infile):
		self.vocab_size = 0
		self.emb_size = opt.emb_size
		self.embeddings = []
		self.w2vvocab = {}
		self.v2wvocab = []
		
		# load pretrained embedings
		with open(infile, 'r') as infile:
			parts = infile.readline().split()
			assert len(parts) == 2
			vs, es = int(parts[0]), int(parts[1])
			assert es == self.emb_size

			# add an UNK token
			self.pretrained = np.empty((vs, es), dtype=np.float)
			self.pt_v2wvocab = []
			self.pt_w2vvocab = {}
			cnt = 0
			for line in infile:
				parts = line.rstrip().split(' ')
				word = parts[0]