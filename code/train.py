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
from train_data import *

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

#vars = ['vocab', 'train_src', 'dev_src', 'test_src', 'train_tgt', 'dev_tgt', 'test_tgt', 'train_src_Q', 'train_tgt_Q', 'train_src_Q_iter', 'train_tgt_Q_iter', 'length']
#for var in vars:
#	if var not in locals() and var not in globals(): print(var, 'not imported '); exit()
#	print(var, 'imported')


#def train(opt):
if __name__ == "__main__":
	"""Train Process:
Require => labeled SOURCE corpus Xsrc; unlabeled TARGET corpus Xtgt; Hyperpamameter λ > 0, k ∈ N, c > 0; Lp(ˆy, y) crossentropy loss.
=> Main iteration
repeat
	=> Q iterations
	for qiter = 1 to k do
		Sample unlabeled batch x_src ~ X_src
		Sample unlabeled batch x_tgt ~ X_tgt
		f_src = F (x_src)
		f_tgt = F (x_tgt) . feature vectors
		lossq = -Q(f_src) + Q(f_tgt)
		Update Q parameters to minimize lossq
		ClipWeights(Q, -c, c)
	
	=> F&P iteration
	Sample labeled batch (x_src, y_src) ~ Xsrc
	Sample unlabeled batch xtgt ~ Xtgt
	f_src = F (x_src)
	f_tgt = F (x_tgt)
	loss = Lp(P(f_src); y_src) + λ * (Q(f_src) - Q(f_tgt))
	Update F , P parameters to minimize loss
until convergence
	"""

	# data
	vocab, train_src, dev_src, test_src, train_tgt, dev_tgt, test_tgt, train_src_Q, train_tgt_Q, train_src_Q_iter, train_tgt_Q_iter, length = get_train_data(opt)

	# models
	log.info('Initializing models...')
	if opt.model.lower() == 'dan': F = DAN_Feature_Extractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn)
	elif opt.model.lower() == 'lstm': F = LSTM_Feature_Extractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout, opt.bdrnn, opt.attn)
	elif opt.model.lower() == 'cnn': F = CNN_Feature_Extractor(vocab, opt.F_layers, opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
	else: raise Exception('Unknown model')

	P = Sentiment_Classifier(opt.P_layers, opt.hidden_size, opt.num_labels, opt.dropout, opt.P_bn)
	Q = Language_Detector(opt.Q_layers, opt.hidden_size, opt.dropout, opt.Q_bn)
	log.info('Done...')

	optimizerFP = optimizers.Adam(lr=opt.learning_rate)		#.(list(F.parameters()) + list(P.parameters()))	# clipvalue = [opt.clip_lower, opt.clip_upper]
	optimizerQ = optimizers.Adam(lr=opt.Q_learning_rate)	#.(Q.parameters())	# clipvalue = [opt.clip_lower, opt.clip_upper]
	
	Q.compile(optimizer=optimizerQ)
	best_acc = 0.0
	# train tgt iterator
	train_tgt_iter = iter(train_tgt)
	""" Main iterations """
	for epoch in trange(opt.epochs):
		F.trainable = True
		P.trainable = True
		Q.trainable = True
		
		# for training accuracy
		correct, total = 0, 0
		sum_src_q, sum_tgt_q = (0, 0.0), (0, 0.0)	# qiter number, loss_q
		grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
		
		# train src iterator
		train_src_iter = iter(train_src)
		for i, (inputs_src, labels_src) in tqdm(enumerate(train_src_iter), total=length['train_src']//opt.batch_size + 1):
			""" sample batches: labeled (xsrc, ysrc) in Xsrc """
			""" sample unlabeled xtgt in Xtgt """
			try:
				inputs_tgt, _ = next(train_tgt_iter)  # tgt labels not used
			except:
				# check if tgt data is exhausted
				train_tgt_iter = iter(train_tgt)
				inputs_tgt, _ = next(train_tgt_iter)
			
			""" Q iterations: """
			q_critic = opt.q_critic
			if q_critic>0 and ((epoch==0 and i<=25) or (i%500==0)): q_critic = 10
			freeze(F.fcnet)
			freeze(P.net)
			unfreeze(Q.net)
			Q.clip_weights()

			for qiter in range(q_critic):
				""" sample unlabeled batches: xsrc in Xsrc, xtgt in Xtgt """
				# get a minibatch of data
				try:
					# labels are not used
					inputs_src_Q, _ = next(train_src_Q_iter)
				except StopIteration:
					# check if dataloader is exhausted
					train_src_Q_iter = iter(train_src_Q)
					inputs_src_Q, _ = next(train_src_Q_iter)
				try:
					inputs_tgt_Q, _ = next(train_tgt_Q_iter)
				except StopIteration:
					train_tgt_Q_iter = iter(train_tgt_Q)
					inputs_tgt_Q, _ = next(train_tgt_Q_iter)
				
				""" extract features : f_src, f_tgt = F(x_src), F(x_tgt) """
				features_src = F(inputs_src_Q)
				features_tgt = F(inputs_tgt_Q)
				
				""" calculate loss_q : loss_q = -Q(f_src) + Q(f_tgt) """

				#"""
				#o_src_ad = Q(features_src)
				#o_tgt_ad = Q(features_tgt)
				
				#l_src_ad = tf.reduce_mean(o_src_ad, axis=-1)	#torch.mean(o_en_ad)
				#l_tgt_ad = tf.reduce_mean(o_tgt_ad, axis=-1)	#torch.mean(o_tgt_ad)
				
				#(-l_src_ad).backward()
				#log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
				#sum_src_q = (sum_src_q[0] + 1, sum_src_q[1] + l_src_ad.item())

				#l_tgt_ad.backward()
				#log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
				#sum_tgt_q = (sum_tgt_q[0] + 1, sum_tgt_q[1] + l_tgt_ad.item())

				#"""

				""" update Q to minimise loss_q """
				l_src_ad = Q.train_step(features_src, 'src')['loss_ad']
				l_tgt_ad = Q.train_step(features_tgt, 'tgt')['loss_ad']

				# summed Q losses
				sum_src_q = (sum_src_q[0] + 1, sum_src_q[1] + l_src_ad)
				sum_tgt_q = (sum_tgt_q[0] + 1, sum_tgt_q[1] + l_tgt_ad)

				""" clip Q weights """
				Q.clip_weights()
			
			""" F&P iteration """
			F.unfreeze()
			P.unfreeze()
			Q.freeze()
			if opt.fix_emb:	freeze(F.emb_layer)

			""" extract features : f_src, f_tgt = F(x_src), F(x_tgt) """
			features_src = F(inputs_src)
			features_tgt = F(inputs_tgt)
			
			""" calculate loss : loss = Lp(P(f_src); y_src) + λ * (Q(f_src) - Q(f_tgt)) """
			o_src_sent = P(features_src)
			l_src_sent = losses.SparseCategoricalCrossentropy(o_src_sent, labels_src)
			#l_src_sent.backward(retain_graph=True)
			
			#o_src_ad, l_src_ad = Q(features_src, compute_loss=True)
			#o_tgt_ad, l_tgt_ad = Q(features_tgt, compute_loss=True)
			
			#(opt.lambd*l_en_ad).backward(retain_graph=True)
			#(-opt.lambd*l_ch_ad).backward()

			l_src_ad = Q.train_step(features_src, 'src', opt._lambda)['loss_ad']
			l_tgt_ad = Q.train_step(features_tgt, 'tgt', opt._lambda)['loss_ad']

			pred = np.max(o_src_sent, axis=1)
			total += len(labels_src) #np.array(targets_src).shape(0)
			correct += np.sum(pred == labels_src)

			##features_en = F(inputs_en)
			##o_en_sent = P(features_en)
			#l_en_sent = functional.nll_loss(o_en_sent, targets_en)
			#l_en_sent.backward(retain_graph=True)
			##o_en_ad = Q(features_en)
			##l_en_ad = torch.mean(o_en_ad)
			#(opt.lambd*l_en_ad).backward(retain_graph=True)
			## training accuracy
			#_, pred = torch.max(o_en_sent, 1)
			#total += targets_en.size(0)
			#correct += (pred == targets_en).sum().item()

			##features_ch = F(inputs_ch)
			##o_ch_ad = Q(features_ch)
			##l_ch_ad = torch.mean(o_ch_ad)
			#(-opt.lambd*l_ch_ad).backward()

			#optimizerFP.step()

		log.info('\n\nl_src_ad = \n' + str(l_src_ad))
		log.info('\n\nl_tgt_ad = \n' + str(l_tgt_ad))


# train.py
#if __name__ == "__main__":
#	train(opt)