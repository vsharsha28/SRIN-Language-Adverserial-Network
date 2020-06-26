# data.py

#!/usr/bin/env ipython

import tensorflow as tf

import json, os, io
from pathlib import Path
from tqdm import tqdm, trange
os.chdir(os.path.dirname(__file__))

from options import opt
from vocab import *

label_dtype = opt.label_dtype

def decode_json(infile, lines=None, reviews_data=None, max_seq_len=None):
	assert os.path.isfile(Path(infile)), str(os.getcwd() / infile) + " doesn't exist, extract_data first"
	with io.open(Path(infile), 'r', encoding='utf-8') as infile:
		if lines is not None and lines > 0: z = zip(trange(lines), infile)
		else: z = enumerate(infile)
		if reviews_data == 'Amazon reviews':
			ret = []
			stars = 0
			for _, line in z:
				dic = json.loads(line)
				ret += [[str(dic["review_title"] + dic["review_body"]).strip().split()[:max_seq_len], tf.cast(int(dic['stars']), dtype=opt.label_dtype)]]
				stars = max(stars, int(dic['stars']))
			return ret, stars
		return [json.loads(line) for line in tqdm(infile.read().strip().split('\n')[:lines])]


def decode_json_iterate(infile, lines=None, reviews_data=None, max_seq_len=None):
	assert os.path.isfile(Path(infile)), str(os.getcwd() / infile) + " doesn't exist, extract_data first"
	with tqdm(total=os.path.getsize(infile) if not lines else lines) as pbar:
		with io.open(Path(infile), 'r', encoding='utf-8') as infile:
			if lines is not None and lines > 0: z = zip(trange(1, lines+1), infile)
			else: z = enumerate(infile)
			for num, line in z:
				if not line or lines is not None and num > lines: break;
				dic = json.loads(line)
				pbar.update(len(line) if not lines else 1)
				yield dic if reviews_data != 'Amazon reviews' \
					else [str(dic["review_title"] + dic["review_body"]).strip().split()[:max_seq_len], tf.cast(int(dic["stars"]), dtype=opt.label_dtype)]


class AmazonReviews:
	"""
	get Amazon reviews data from the extracted data => review title + review body + eos_tok : stars
parameters:
	path : str => path to 'Amazon reviews' directory with '/' as separator
	eos_tok : str
	max_seq_len : int
	"""
	def __init__(self, path:str=None, eos_tok=opt.eos_tok, max_seq_len=opt.max_seq_len, star_rating=5):
		super(AmazonReviews, self).__init__()
		self.path = Path('Amazon reviews') if not path else Path(path)
		self.dats = {}
		self.dats['train'] = self.path / 'train'
		self.dats['dev'] = self.path / 'dev'
		self.dats['test'] = self.path / 'test'
		self.eos_tok = eos_tok
		self.max_seq_len = max_seq_len
		opt.labels = self.star_rating = star_rating


	def load_data(self, lang, dat, lines=-1):
		"""
		load all data in one go
parametrs:
		lang : str => de, en, es, fr, ja, zh
		dat : str => train, dev, test
		lines : int
return:
		tuple of
			dataset of reviews (split) and their star ratings
			max_seq_length
		"""
		infile = self.dats[dat] / str('dataset_' + lang + '_' + dat + '.json')
		data, self.star_rating = decode_json(infile, lines=lines, reviews_data='Amazon reviews', max_seq_len=self.max_seq_len)
		return data


	def load_data_generator(self, lang, dat, lines=-1):
		"""
		iterate over the data file line by line (for less RAM devices like this one - not completely implemented)
parametrs:
		lang : str => de, en, es, fr, ja, zh
		dat : str => train, dev, test
		lines : int
yield:
		generator witch generates a list of one review and its star rating at a time
		"""
		infile = self.dats[dat] / str('dataset_' + lang + '_' + dat + '.json')
		return decode_json_iterate(infile, lines=lines, reviews_data='Amazon reviews', max_seq_len=self.max_seq_len)


if __name__ == "__main__":
	infile = Path('Amazon reviews/test/dataset_en_test.json')
	#assert os.path.isfile(infile), str(os.getcwd() / infile) + " doesn't exist"
	#for x in decode_json_iterate(infile, 30): print(x)
	#print(decode_json(infile, 30))
	vocab = Vocab(opt.pre_trained_src_emb_file)
	rev = AmazonReviews()
	data = rev.load_data(lang='en', dat='train', lines=100)
	data = vocab.pad_sequences(data)
	print('\n', data)
	for dat in data.take(3):
		print(dat)
	#sequence_input = layers.InputLayer(input_shape=(opt.max_seq_len,), dtype='int32')(np.array([x for x, y in data.as_numpy_iterator()], dtype='int32'))
	emb_layer = vocab.init_embed_layer()
	#emb_layer(sequence_input)
	print(emb_layer(tf.convert_to_tensor([x for x, y in data.as_numpy_iterator()], dtype='int32')))