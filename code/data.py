#!/usr/bin/env ipython

import tensorflow as tf
import json, os, io
from pathlib import Path
os.chdir(os.path.dirname(__file__))

from options import opt

def decode_json(infile, lines=None, reviews_data=None):
	assert os.path.isfile(Path(infile)), str(os.getcwd() / infile) + " doesn't exist, extract_data first"
	with io.open(Path(infile), encoding='utf-8') as infile:
		data = infile.read().strip().split('\n')[:lines]
		return [json.loads(line) for line in data] if reviews_data is not 'Amazon reviews' else [[dic['review_body'], dic['stars']]  for x in data for dic in [json.loads(x)]] #dic["review title"]


def decode_json_iterate(infile, lines=None):
	assert os.path.isfile(Path(infile)), str(os.getcwd() / infile) + " doesn't exist, extract_data first"
	with open(Path(infile), 'r') as infile:
		for num, line in enumerate(infile.readlines(), 1):
			if not line or lines != None and num > lines: break;
			dic = json.loads(line)
			yield dic if reviews_data is not 'Amazon reviews' else tf.data.Dataset.from_tensor_slices([dic["review_body"], dic["stars"]]) #dic["review title"]


class AmazonReviews(tf.data.Dataset):
	"""
	get Amazon reviews data from the extracted data
parameters:
	path : str => path to 'Amazon reviews' directory with '/' as separator
	reviews_data => 
	"""
	def __init__(self, path:str=None, reviews_data='Amazon reviews'):
		super(AmazonReviews, self),__init__()
		self.path = Path('Amazon reviews') if not path else Path(path)
		self.dats = {}
		self.dats['train'] = self.path / 'train'
		self.dats['dev'] = self.path / 'dev'
		self.dats['test'] = self.path / 'test'


	def load_data(self, lang, dat, lines=-1):
		"""
		load all data in one go
parametrs:
		lang : str => de, en, es, fr, ja, zh
		dat : str => train, dev, test
		lines : int
return:
		list of reviews and their star ratings
		"""
		infile = self.dats[dat] / str('dataset_' + lang + '_' + dat + '.json')
		return tf.data.Dataset.from_tensor_slices(decode_json(infile, lines=lines, reviews_data='Amazon reviews'))


	def load_data_generator(self, lang, dat, lines=-1):
		"""
		iterate over the data file line by line (for less RAM devices like this one)
parametrs:
		lang : str => de, en, es, fr, ja, zh
		dat : str => train, dev, test
		lines : int
yield:
		generator witch generates a list of one review and its star rating at a time
		"""
		infile = self.dats[dat] / str('dataset_' + lang + '_' + dat + '.json')
		return decode_json_iterate(infile, lines=lines, reviews_data='Amazon reviews')	#tf.data.Dataset.from_generator



if __name__ == "__main__":
	infile = Path('Amazon reviews/test/dataset_en_test.json')
	assert os.path.isfile(infile), str(os.getcwd() / infile) + " doesn't exist"
	for x in decode_json_iterate(infile, 3): print(x)
	print(decode_json(infile, 3))