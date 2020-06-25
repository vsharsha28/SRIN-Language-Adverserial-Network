# train_data.py

#!/usr/bin/env ipython

#import torch
#import torch.nn as nn
#import torch.nn.functional as functional
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from torchnet.meter import ConfusionMeter


import tensorflow as tf
from tensorflow.keras import optimizers, losses
import json

import os, random, sys, logging, argparse
from tqdm import tqdm
from pathlib import Path
os.chdir(os.path.dirname(__file__))

from options import *
from data import *
from vocab import *
from utils import *
from models import *

#tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(True)

#random.seed(opt.random_seed)
#torch.manual_seed(opt.random_seed)

# save logs
if not os.path.exists(opt.model_save_file): os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
 
# output options
log.info('Training ADAN with options:')
log.info(opt)


def get_train_data(opt):

	# vocab
	log.info(f'Loading Embeddings...')
	vocab = Vocab(opt.pre_trained_src_emb_file)
	vocab = Vocab(opt.pre_trained_tgt_emb_file)
	log.info(f'Done.')

	# datasets
	length = {}

	# src_lang datasets
	log.info(f'Loading src datasets...')
	reviews_src_obj = AmazonReviews(path=opt.data_path, max_seq_len=opt.max_seq_len)
	train_src = reviews_src_obj.load_data(lang=opt.src_lang, dat='train', lines=opt.train_size_src)
	dev_src = reviews_src_obj.load_data(lang=opt.src_lang, dat='dev', lines=1000)
	test_src = reviews_src_obj.load_data(lang=opt.src_lang, dat='test', lines=1000)
	length['train_src'], length['dev_src'], length['test_src'] = len(train_src), len(dev_src), len(test_src)
	log.info('Done loading src datasets.')
 
	# tgt_lang datasets
	log.info(f'Loading tgt datasets...')
	reviews_tgt_obj = AmazonReviews(path=opt.data_path, max_seq_len=opt.max_seq_len)
	train_tgt = reviews_tgt_obj.load_data(lang=opt.tgt_lang, dat='train', lines=opt.train_size_tgt)
	dev_tgt = reviews_tgt_obj.load_data(lang=opt.tgt_lang, dat='dev', lines=1000)
	test_tgt = reviews_tgt_obj.load_data(lang=opt.tgt_lang, dat='test', lines=1000)
	length['train_tgt'], length['dev_tgt'], length['test_tgt'] = len(train_tgt), len(dev_tgt), len(test_tgt)
	log.info('Done loading tgt datasets.')

	opt.num_labels = max(reviews_src_obj.star_rating, reviews_tgt_obj.star_rating)
	if opt.max_seq_len < 0 or not opt.max_seq_len:
		maxlen_src, maxlen_tgt = max(list(len(x) for x in train_src)), max(list(len(x) for x in train_tgt))
		opt.max_seq_len = max(maxlen_src, maxlen_tgt)
	del reviews_src_obj, reviews_tgt_obj

	# pad src datasets (-> Dataset)
	log.info('Padding src datasets...')
	train_src = vocab.pad_sequences(train_src, max_len=opt.max_seq_len)
	dev_src = vocab.pad_sequences(dev_src, max_len=opt.max_seq_len)
	test_src = vocab.pad_sequences(test_src, max_len=opt.max_seq_len)
	log.info('Done padding tgt datasets...')

	# pad tgt datasets (-> Dataset)
	log.info('Padding tgt datasets...')
	train_tgt = vocab.pad_sequences(train_tgt, max_len=opt.max_seq_len)
	dev_tgt = vocab.pad_sequences(dev_tgt, max_len=opt.max_seq_len)
	test_tgt = vocab.pad_sequences(test_tgt, max_len=opt.max_seq_len)
	log.info('Done padding tgt datasets...')

	# dataset loaders
	log.info('Shuffling and batching...')
	train_src = train_src.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	train_tgt = train_tgt.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	train_src_Q = tf.identity(train_src)
	train_tgt_Q = tf.identity(train_src)
	train_src_Q_iter = iter(train_src_Q)
	train_tgt_Q_iter = iter(train_tgt_Q)
	
	dev_src = dev_src.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	dev_tgt = dev_tgt.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	
	test_src = test_src.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	test_tgt = test_tgt.shuffle(buffer_size=opt.batch_size*10, reshuffle_each_iteration=True).batch(opt.batch_size)
	log.info('Done shuffling and batching.')

	return vocab, train_src, dev_src, test_src, train_tgt, dev_tgt, test_tgt, train_src_Q, train_tgt_Q, train_src_Q_iter, train_tgt_Q_iter, length


#vocab, train_src, dev_src, test_src, train_tgt, dev_tgt, test_tgt, train_src_Q, train_tgt_Q, train_src_Q_iter, train_tgt_Q_iter, length = get_train_data(opt)
#imported = True