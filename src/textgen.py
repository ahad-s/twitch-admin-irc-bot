from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper, MultiRNNCell

import tensorflow as tk

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import load
from nltk.corpus import stopwords
from nltk import word_tokenize
import itertools


sent_start = "SENT_START"
sent_end = "SENT_END"
not_in_vocab_token = "WHODISNEWTOKEN"
vocab_num = 1000 # todo fix depending on dataset size
max_sent_len = 10

class DataStream(object):
	def __init__(self, X, Y, batch_size):
		assert X.shape[0] == Y.shape[0]
		self.x = X
		self.y = Y
		self.size = X.shape[0]
		self.batch_size = batch_size
		self.batch_start = 0
		self.randomize_data()

	def get_next_batch(self):
		X = self.x[batch_start:min(batch_start+batch_size, self.size-1), :]
		Y = self.y[batch_start:min(batch_start+batch_size, self.size-1), :]
		batch_start += batch_size
		return X, Y

	def randomize_data(self):
		# shuffle x, y with the same order
		rng_state = np.random.get_state()
		np.random.shuffle(self.x)
		np.random.set_state(rng_state)
		np.random.shuffle(self.y)


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

	pad_list = lambda l, maxlen: l + ([-1]*(maxlen - len(l))) \
						if len(l) < maxlen else l

	# make sure all lists are of length 15, with padding of 0s (= arbitrary?)
	x_sents = [pad_list([word_to_idx[w] for w in sent], max_sent_len) 
				for sent in tok_sents if len(sent) > 0 and len(sent) <= max_sent_len]
	y_sents = [pad_list([word_to_idx[w] for w in sent[1:]], max_sent_len-1) 
				for sent in tok_sents if len(sent) > 0 and len(sent) <= max_sent_len]

	x_train = np.asarray(x_sents)
	y_train = np.asarray(y_sents)

	print y_train.shape
	print "----------------"
	print x_train.shape


# one-hot encoded sentences
# note: our seq2seq is always given correct inputs even with bad prediction
def skynet(x, y):

	sess = tf.Session()

	HIDDEN_NEURONS = 100
	NUM_LAYERS = 3
	max_in_size = max_sent_len
	max_out_size = max_sent_len - 1
	in_vocab_size = vocab_num
	out_vocab_size = vocab_num
	word_emb_dim = 256
	batch_size = 128 # test this out

	dropout = tf.placeholder(tf.float)

	encode_in = [tf.placeholder(tf.int32, shape=(None, ), name="ei_%i" %i)
					for i in xrange(max_in_size)]
	labels = [tf.placeholder(tf.int32, shape=(None, ), name="lab_%i" %i)
				for i in xrange(max_out_size)]

	# decode_in must be shifted one right 
	decode_in = [tf.zeros_like(encode_in[0], dtype=np.int32, name="GO")] + \
					labels[:-j1]

	keep_probability = tf.placeholder("float")

	cells = [DropoutWrapper(BasicLSTMCell(word_emb_dim), 
			output_keep_prob=keep_probability) 
			for i in range(3)]

	stack_lstm = MultiRNNCell(cells)

	with tf.variable_scape("decoders") as scope:
		decode_out, decode_state = seq2seq.embedding_rnn_seq2seq(
			encode_in, decode_in, stack_lstm, max_in_size, max_out_size)

		scope.reuse_variables()

		decode_out_test, decode_state_test = seq2seq.embedding_rnn_seq2seq(
			encode_in, decode_in, stack_lstm, max_in_size, max_out_size)

	loss_weights = [tf.ones_like(lab, dtype=tf.float32) for lab in labels]
	loss = seq2seq.sequence_loss(decode_out, labels, loss_weights, max_out_size)
	opt = tf.train.AdamOptimizer(1e-4)
	train_op = opt.minimize(loss)

	sess.run(tf.initialize_all_variables())

	# TRAINING STUFF #

	# given x, y, convert for input into tf
	def get_feed_dict(x, y):
		feed = {encode_in[i]: x[i] for i in xrange(max_in_size)}
		feed.update({labels[i]: y[i] for i in xrange(max_in_size)})
		return feed

	def train_batch(data_stream):
		X, Y = data_stream.get_next_batch()
		feed_dict = get_feed_dict(X, Y)
		feed_dict[keep_prob] = 0.5
		_, out = sess.run([train_op, loss], feed_dict)

	def get_eval_batch_data(data_stream):
		X, Y = data_stream.get_next_batch()
		feed_dict = get_feed_dict(X, Y)
		feed_dict[keep_prob] = 1
		all_output = sess.run([loss] + decode_out_test, feed_dict)
		eval_loss = all_output[0]
		decode_out = np.array(all_output[1:].tranpose([1, 0, 2]))
		return eval_loss, decode_out, X, Y

	def eval_batch(data_stream, num_batch):
		losses = []
		pred_loss = []
		for i in xrange(num_batch):
			eval_loss, output, X, Y = get_eval_batch_data(data_stream)
			losses.append(eval_loss)

			for idx in xrange(len(output)):
				real = Y.T[idx]
				pred = np.argmax(output, axis=2)[idx]
				pred_loss.append(all(real==predict))
		return np.mean(losses), np.mean(pred_loss)

	train_stream = DataStream(x, y, 50)
	epochs = 1000
	check_every = 50

	for i in xrange(epochs):
		try:
			train_batch(train_stream)
		except KeyboardInterrupt:
			print "INTERRUPTED BY FAM"

	# TESTING STUFF - todo get more data#

	# eval_loss, output, X, Y = get_eval_batch_data(test_stream)



# text_to_sentences()