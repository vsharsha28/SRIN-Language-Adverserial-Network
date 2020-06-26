# utils.py

#!/usr/bin/env ipython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing

import pdb
import numpy as np

from options import *

def freeze(net):
	net.trainable = False

def unfreeze(net):
	net.trainable = True

def argmax32(arr, axis=-1, dtype=opt.label_dtype):
	return tf.cast(np.argmax(arr, axis=-1), dtype=dtype)

def pad(x : list, y : list, eos_idx : int, sort:bool=False):
	#inputs, lengths = zip(*x)
	inputs = x
	lengths = [len(l) for l in x]
	max_len = max(lengths)
	# pad sequences
	padded_inputs = tf.fill((len(inputs), max_len), eos_idx, dtype=tf.int64)
	for i, row in enumerate(inputs):
		assert eos_idx not in row, f'EOS in sequence {row}'
		padded_inputs[i][:len(row)] = tf.convert_to_tensor(row, dtype=tf.int64)
	lengths = tf.convert_to_tensor(lengths, dtype=tf.int64)
	y = tf.reshape(tf.convert_to_tensor(y, dtype=tf.int64), -1)
	if sort:
		# sort by length
		sorted_lengths = lengths.sort(axis=0, direction='DESCENDING')
		sorting_idx = keras.backend.eval(sorted_lengths)
		padded_inputs = tf.gather(params=padded_inputs, indices=sorting_idx, axis=0)
		y = tf.gather(params=y, indices=sorting_idx, axis=0)
		return (padded_inputs, sorted_lengths), y
	else:
		return (padded_inputs, lengths), y


def my_collate(batch : list, sort : bool):
	x, y = zip(*batch)
	with tf.device(opt.device):
		x, y = pad(x, y, opt.eos_idx, sort)
	return (x, y)


def sorted_collate(batch):
	return my_collate(batch, sort=True)

def unsorted_collate(batch):
	return my_collate(batch, sort=False)


if __name__ == "__main__":
	pass