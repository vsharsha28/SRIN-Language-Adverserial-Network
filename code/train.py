#!/usr/bin/env ipython

import tensorflow as tf
import json

import os
from pathlib import Path
os.chdir(os.path.dirname(__file__))

def decode(infile, n=-1):
	assert os.path.isfile(infile), str(os.getcwd() / infile) + " doesn't exist"
	with open(infile, 'r') as infile:
		if n == -1:
			while(not infile.):
				line = infile.readline()
				dic = json.loads(line)
				yield(tf.convert_to_tensor([dic['review_body'], dic['stars']]))
		else


infile = Path('Amazon Reviews/test/dataset_en_test.json')
assert os.path.isfile(infile), str(os.getcwd() / infile) + " doesn't exist"
with open(infile, 'r') as infile:
	line = infile.readline()
	dic = json.loads(line)
	print(tf.convert_to_tensor([dic['review_body'], dic['stars']]))
