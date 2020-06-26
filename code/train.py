#!/usr/bin/env ipython
# foo = open('train.py', 'r'); foo.readline(); exec(foo.read()); foo.close()
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

	optimizer_FP = Optimizer_FP(models=[F, P, Q], lr=opt.learning_rate, clip_lim=opt.clip_lim_FP)
	if not opt.clip_Q: optimizer_Q = optimizers.Adam(lr=opt.Q_learning_rate)
	else: optimizer_Q = optimizers.Adam(lr=opt.Q_learning_rate, clipvalue=opt.clipvalue)
	
	F.compile(optimizer=optimizer_FP)
	P.compile(optimizer=optimizer_FP)
	Q.compile(optimizer=optimizer_Q)

	best_acc = 0.0
	# train tgt iterator
	train_tgt_iter = iter(train_tgt)
	log.info('Main Iteration begin...')
	""" Main iterations """
	for epoch in trange(opt.epochs):
		F.unfreeze()
		P.unfreeze()
		Q.unfreeze()
		F.freeze_emb_layer()
		
		# for training accuracy
		correct, total = 0, 0
		sum_src_q, sum_tgt_q = (0, 0.0), (0, 0.0)	# qiter number, loss_q
		grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
		
		# train src iterator
		train_src_iter = iter(train_src)
		log.info('Q iteration begin...')
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
			F.freeze()
			P.freeze()
			Q.unfreeze()
			F.freeze_emb_layer()
			#Q.clip_weights()

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
				""" update Q to minimise loss_q """
				l_src_ad = Q.train_step(features_src, 'src')['loss']
				l_tgt_ad = Q.train_step(features_tgt, 'tgt')['loss']
				# summed Q losses
				sum_src_q = (sum_src_q[0] + 1, sum_src_q[1] + l_src_ad)
				sum_tgt_q = (sum_tgt_q[0] + 1, sum_tgt_q[1] + l_tgt_ad)

				""" clip Q weights """
				Q.clip_weights()
			
			log.info('Q iteration done.')

			""" F&P iteration """
			F.unfreeze()
			P.unfreeze()
			Q.freeze()
			if opt.fix_emb:	pass #F.freeze_emb_layer()
			elif epoch>3: F.unfreeze_emb_layer()

			""" extract features : f_src, f_tgt = F(x_src), F(x_tgt) """
			""" calculate loss : loss = Lp(P(f_src); y_src) + λ * (Q(f_src) - Q(f_tgt)) """
			metrices = optimizer_FP.call(inputs_src, inputs_tgt, labels_src, labels_tgt=None, _lambda=opt._lambda, supervised=False)
			#pred = argmax32(o_src_sent)
			#total += len(labels_src)
			#correct += np.sum(pred == labels_src)

		#log.info('\n\nl_src_ad = \n' + str(l_src_ad))
		#log.info('\n\nl_tgt_ad = \n' + str(l_tgt_ad))
		log.info(f'\n\n result :\n' + str(metrices))

	log.info('\nMain iteration done.')
	log.info(f' (Q(features_src) < Q(features_tgt)) : {np.sum(Q(features_src) < Q(features_tgt))}')
	log.info(f' (Q(features_src) > Q(features_tgt)) : {np.sum(Q(features_src) > Q(features_tgt))}')
	log.info(f' Q precision in differentiating src-tgt : {np.sum(Q(features_src) > Q(features_tgt)) / (np.sum(Q(features_src) < Q(features_tgt)) + np.sum(Q(features_src) > Q(features_tgt)))}')
	log.info(f' Q accuracy : unknown')
	log.info(f'\n\n RESULT :\n' + str(metrices))

# train.py
#if __name__ == "__main__":
#	train(opt)