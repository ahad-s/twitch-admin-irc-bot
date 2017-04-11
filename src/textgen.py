# import tensorflow as tk
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import load
from nltk.corpus import stopwords
from nltk import word_tokenize
import itertools

class TextGenerator(object):
	def __init__(self, train_file="esea.txt"):
		pass


sent_start = "SENT_START"
sent_end = "SENT_END"
not_in_vocab_token = "WHODISNEWTOKEN"
vocab_num = 1000 # todo fix?

def text_to_sentences(fname = "esea.txt"):
	sent_maker = load('tokenizers/punkt/english.pickle')
	f = open(fname, "rb+")
	sents = map(lambda s: s.rstrip().decode("utf-8"), f.readlines())
	l = lambda s: [sent_start] + list(s) + [sent_end]
	sents_broken = [x for s in sents for x in sent_maker.tokenize(s)]
	tok_sents = [word_tokenize(s) for s in sents_broken]

	# only get top x frequent
	wfreq_dist = nltk.FreqDist(itertools.chain(*tok_sents)) # .chain --> combines lists
	vocab = wfreq_dist.most_common(vocab_num)

	# one-hot encoding vector
	idx_word = [x[0] for x in vocab] + [not_in_vocab_token]
	word_to_idx = dict([(w, i) for i, w in enumerate(idx_word)])

	for i, sent in enumerate(tok_sents):
		tok_sents[i] = [w if w in word_to_idx else not_in_vocab_token 
						for w in sent]

	# y is used to predict the "next word"
	x_sents = [[word_to_idx[w] for w in sent[:-1]] for sent in tok_sents]
	y_sents = [[word_to_idx[w] for w in sent[1:]] for sent in tok_sents]

	x_train = np.asarray(x_sents)
	y_train = np.asarray(y_sents)

	print y_train
	print "----------------"
	print x_train


text_to_sentences()
	


