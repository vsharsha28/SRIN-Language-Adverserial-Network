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
	#train(opt)
	if True:
	# end of epoch
		log.info('Ending epoch {}'.format(epoch+1))
		# logs
		if sum_en_q[0] > 0:
			log.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
			log.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
		# evaluate
		log.info('Training Accuracy: {}%'.format(100.0*correct/total))
		log.info('Evaluating English Validation set:')
		evaluate(opt, yelp_valid_loader, F, P)
		log.info('Evaluating Foreign validation set:')
		acc = evaluate(opt, chn_valid_loader, F, P)
		if acc > best_acc:
			log.info(f'New Best Foreign validation accuracy: {acc}')
			best_acc = acc
			torch.save(F.state_dict(),
					'{}/netF_epoch_{}.pth'.format(opt.model_save_file, epoch))
			torch.save(P.state_dict(),
					'{}/netP_epoch_{}.pth'.format(opt.model_save_file, epoch))
			torch.save(Q.state_dict(),
					'{}/netQ_epoch_{}.pth'.format(opt.model_save_file, epoch))
		log.info('Evaluating Foreign test set:')
		evaluate(opt, chn_test_loader, F, P)
	log.info(f'Best Foreign validation accuracy: {best_acc}')