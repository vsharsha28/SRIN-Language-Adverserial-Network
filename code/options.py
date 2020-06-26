# options.py

#!/usr/bin/env ipython

import tensorflow as tf

import os, sys, logging, argparse
from pathlib import Path
os.chdir(os.path.dirname(__file__))

sys.argv = ['./'+__file__, '--learning_rate', '1', '--Q_learning_rate', '1', '--clipvalue', '10']	# uncomment when in notebook
parser = argparse.ArgumentParser()

# dataset arguments
parser.add_argument('--data_path', default=None)
parser.add_argument('--src_lang', default='en')
parser.add_argument('--tgt_lang', default='fr')
parser.add_argument('--train_size_src', type=int, default=None)		# use all
parser.add_argument('--train_size_tgt', type=int, default=None)		# use all
parser.add_argument('--num_labels', type=int, default=5+1)			# max reviews rating
parser.add_argument('--iterate', action='store_true')				# read through iterations
parser.add_argument('--label_dtype', default=tf.int32)

# sequences and vocab arguments
parser.add_argument('--max_seq_len', type=int, default=50)			# None for no truncate
parser.add_argument('--unk_tok', type=str, default='<unk>')
parser.add_argument('--bos_tok', type=str, default='<s>')
parser.add_argument('--eos_tok', type=str, default='</s>')

# training arguments
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--model_save_file', default='./saved_models/adan')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--Q_learning_rate', type=float, default=0.0005)

# bwe arguments
parser.add_argument('--emb_filename', default='')
parser.add_argument('--n_vecs', type=int, default=20000)
parser.add_argument('--random_emb', action='store_true')
parser.add_argument('--fix_unk', action='store_true')				# use a fixed <unk> token for all words without pretrained embeddings when building vocab
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--pre_trained_src_emb_file', type=str, default='bwe/vectors/wiki.multi.en.vec')
parser.add_argument('--pre_trained_tgt_emb_file', type=str, default='bwe/vectors/wiki.multi.fr.vec')

# Feature Extractor
parser.add_argument('--model', default='dan')						# dan or lstm or cnn
parser.add_argument('--fix_emb', action='store_true')
# for LSTM model
parser.add_argument('--attn', default='dot')						# attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--bidir_rnn', dest='bidir_rnn', action='store_true', default=True)		# bi-directional LSTM
parser.add_argument('--sum_pooling/', dest='avg_pooling', action='store_false')
parser.add_argument('--avg_pooling/', dest='avg_pooling', action='store_true')
# for CNN model
parser.add_argument('--kernel_num', type=int, default=400)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])

# for layers and all models
parser.add_argument('--F_layers', type=int, default=2)
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--Q_layers', type=int, default=2)

parser.add_argument('--q_critic', type=int, default=5)	# Q iterations
parser.add_argument('--_lambda', type=float, default=0.01)

parser.add_argument('--F_bn/', dest='F_bn', action='store_true')
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--P_bn/', dest='P_bn', action='store_true', default=True)
parser.add_argument('--no_P_bn/', dest='P_bn', action='store_false')
parser.add_argument('--Q_bn/', dest='Q_bn', action='store_true', default=True)
parser.add_argument('--no_Q_bn/', dest='Q_bn', action='store_false')

parser.add_argument('--hidden_size', type=int, default=900)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--linear_activation', type=str, default='linear')

parser.add_argument('--clip_Q', type=bool, default=False)
parser.add_argument('--clipvalue', type=float, default=0.01)
parser.add_argument('--clip_lim_FP', type=float, default=None)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')

opt = parser.parse_args()

if not tf.config.list_physical_devices('GPU'):
	opt.device = 'CPU'

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(Path(opt.model_save_file) / 'log.txt')
log.addHandler(fh)

if __name__ == "__main__":
	print("src_embeddings: ", opt.pre_trained_src_emb_file)
	print("tgt_embeddings: ", opt.pre_trained_tgt_emb_file)
	print("debugging: ", opt.debug)
	log.info('start...')