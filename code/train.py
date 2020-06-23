#!/usr/bin/env ipython

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter


import tensorflow as tf
import json

import os, random, sys, logging, argparse
from tqdm import tqdm
from pathlib import Path
os.chdir(os.path.dirname(__file__))

from options import opt
from data import *
from vocab import Vocab
import utils
from models import *

tf.logging.set_verbosity(tf.logging.INFO)

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# save logs
if not os.path.exists(opt.model_save_file):
	os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
 
# output options
log.info('Training ADAN with options:')
log.info(opt)


def train(opt):
	"""Train Process:
repeat
	#Q iterations
	for qiter = 1 to k do
		Sample unlabeled batch x_src âˆ¼ X_src
		Sample unlabeled batch x_tgt âˆ¼ X_tgt
		f_src = F (x_src)
		f_tgt = F (x_tgt) . feature vectors
		lossq = âˆ’Q(f_src) + Q(f_tgt)
		#Update Q parameters to minimize lossq
		ClipWeights(Q, âˆ’c, c)
#Main iteration
	Sample labeled batch (x_src, y_src) âˆ¼ Xsrc
	Sample unlabeled batch xtgt âˆ¼ Xtgt
	f_src = F (x_src)
	f_tgt = F (x_tgt)
	loss = Lp(P(f_src); y_src) + Î»(Q(f_src) âˆ’ Q(f_tgt))
	Update F , P parameters to minimize loss
until convergence
	"""
	F = LSTMFeatureExtractor(vocab, F_options)
	P = SentimentClassifier(P_options)
	Q = LangDetector(Q_options)
	
	optimizer = tf.nn.Adam(list(F.parameters()) + list(P.parameters()), lr=opt.learning_rate)
	optimizerQ = tf.nn.Adam(Q.parameters(), lr=opt.Q_learning_rate)
	
	best_acc = 0.0
	for epoch in range(options[epochs]):
		F.train()
		P.train()
		Q.train()
		
		# for training accuracy
		correct, total = 0, 0
		sum_src_q, sum_tgt_q = (0, 0.0), (0, 0.0)	# qiter number, loss_q
		grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
		
		train_iter_src = iter(train_generator_src)
		for i, (inputs_src, labels_src) in tqdm(enumerate(train_iter_src), total=len(train_src)//options.batch_size):
			# sample batches: labeled (xsrc, ysrc) in Xsrc
			# sample unlabeled xtgt in Xtgt
			try:
				inputs_tgt, _ = next(train_iter_tgt)  # Chinese labels are not used
			except:
				# check if Chinese data is exhausted
				train_iter_tgt = iter(train_generator_tgt)
				inputs_tgt, _ = next(train_iter_tgt)
			
			# Q iterations:
			for qiter in range(options[k]):
				# sample unlabeled batches: xsrc in Xsrc, xtgt in Xtgt
				
				# clip Q weights
				for p in Q.parameters():
					p.data.clamp_(options[lower_lim], options[clip_upper_lim])
				Q.zero_grad()	??
				# get a minibatch of data
				try:
					# labels are not used
					inputs_src_Q, _ = next(train_iter_src_Q)
				except StopIteration:
					# check if dataloader is exhausted
					train_iter_src_Q = iter(train_generator_src_Q)
					inputs_src_Q, _ = next(train_iter_src_Q)
				try:
					inputs_tgt_Q, _ = next(train_iter_tgt_Q)
				except StopIteration:
					train_iter_tgt_Q = iter(train_generator_tgt_Q)
					inputs_tgt_Q, _ = next(train_iter_tgt_Q)
				
				# extract features
				# f_src, f_tgt = F(x_src), F(x_tgt)
				features_src = F(inputs_src_Q)
				features_tgt = F(inputs_tgt_Q)
				
				# calculate loss_q
				# loss_q = âˆ’Q(f_src) + Q(f_tgt)
				o_src_ad = Q(features_en)
				o_tgt_ad = Q(features_tgt)
				
				l_src_ad = torch.mean(o_en_ad)
				l_tgt_ad = torch.mean(o_tgt_ad)
				
				(-l_src_ad).backward()
				l_tgt_ad.backward()
				
				sum_src_q = (sum_src_q[0] + 1, sum_src_q[1] + l_src_ad.item())
				sum_tgt_q = (sum_tgt_q[0] + 1, sum_tgt_q[1] + l_tgt_ad.item())
				
				# update Q to minimise loss_q
				optimizerQ.step()
				
			# extract features
			# f_src, f_tgt = F(x_src), F(x_tgt)
			features_src = F(inputs_src)
			features_tgt = F(inputs_tgt)
			
			# calculate loss
			# loss = Lp(P(f_src); y_src) + Î»(Q(f_src) âˆ’ Q(f_tgt))
			o_src_sent = P(features_src)
			l_src_sent = functional.nll_loss(o_src_sent, labels_src)
			l_src_sent.backward(retain_graph=True)
			
			o_src_ad = Q(features_src)
			o_tgt_ad = Q(features_tgt)
			
			l_src_ad = torch.mean(o_src_ad)
			(options[lambd] * l_src_ad).backward(retain_graph=True)
			
			l_tgt_ad = torch.mean(o_tgt_ad)
			(-options[lambd] * l_tgt_ad).backward()
			
			# training accuracy
			_, pred = torch.max(o_src_sent, 1)
			total += labels_src.size(0)
			correct += (pred == labels_src).sum().item()
			
			# update F , P parameters to minimize loss
			optimizer.step()
			
			

def evaluate(opt, loader, F, P):
	F.eval()
	P.eval()
	it = iter(loader)
	correct = 0
	total = 0
	confusion = ConfusionMeter(opt.num_labels)
	with torch.no_grad():
		for inputs, targets in tqdm(it):
			outputs = P(F(inputs))
			_, pred = torch.max(outputs, 1)
			confusion.add(pred.data, targets.data)
			total += targets.size(0)
			correct += (pred == targets).sum().item()
	accuracy = correct / total
	log.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
	log.debug(confusion.conf)
	return accuracy


if __name__ == '__main__':
	train(opt)
