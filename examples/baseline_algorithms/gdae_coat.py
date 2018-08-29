import os
import shutil
import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import bottleneck as bn

from sklearn import metrics

flags = tf.flags
flags.DEFINE_integer('p_dim', 100, '')
flags.DEFINE_float('lam', 0.01, '')
flags.DEFINE_float('lr', 1e-3, '')
FLAGS = flags.FLAGS

DATA_DIR = '/home/xiaojie/Downloads/coat'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')

class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.weight_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):
        self.construct_weights()

        saver, logits = self.forward_pass()

        # per-user average negative log-likelihood
        # log_softmax_var = tf.nn.log_softmax(logits)
        # neg_ll = -tf.reduce_mean(tf.reduce_sum(
        #     log_softmax_var * self.input_ph, axis=1))

        neg_ll = -0.5 * tf.reduce_mean(tf.reduce_sum(
            self.weight_ph * (self.input_ph - logits) ** 2, axis=1))

        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])

unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())
n_items = len(unique_sid)

def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((tp['rating'],
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

tr_data = load_train_data(os.path.join(pro_dir, 'train.csv'))

def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((tp_tr['rating'],
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((tp_te['rating'],
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                           os.path.join(pro_dir, 'validation_te.csv'))

N = tr_data.shape[0]
idxlist = list(range(N))
# training batch size
batch_size = 8
batches_per_epoch = int(np.ceil(float(N) / batch_size))
N_vad = vad_data_tr.shape[0]
idxlist_vad = list(range(N_vad))
# validation batch size (since the entire validation set might not fit into GPU memory)
batch_size_vad = 290
# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2

n_epochs = 20

p_dims = [FLAGS.p_dim, n_items]
missing_w, observe_w = 0.01, 1.00
tf.reset_default_graph()
dae = MultiDAE(p_dims, lam=FLAGS.lam, lr=FLAGS.lr)
saver, logits_var, loss_var, train_op_var, merged_var = dae.build_graph()

arch_str = "I-%s-I" % ('-'.join([str(d) for d in dae.dims[1:-1]]))

log_dir = 'log/DAE/{}'.format(arch_str)
log_dir = os.path.join(DATA_DIR, log_dir)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

chkpt_dir = 'chkpt/DAE/{}'.format(arch_str)
chkpt_dir = os.path.join(DATA_DIR, chkpt_dir)
if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir) 
print("chkpt directory: %s" % chkpt_dir)

maes_vad = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    best_mae = -np.inf
    
    for epoch in range(n_epochs):
        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = tr_data[idxlist[st_idx:end_idx]]
            
            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
            W = np.ones_like(X) * missing_w
            W[X > 0] = observe_w
            
            feed_dict = {dae.input_ph: X, 
                         dae.weight_ph: W,
                         dae.keep_prob_ph: 0.5}        
            sess.run(train_op_var, feed_dict=feed_dict)

            if bnum % 100 == 0:
                summary_train = sess.run(merged_var, feed_dict=feed_dict)
                summary_writer.add_summary(summary_train, global_step=epoch * batches_per_epoch + bnum) 
                    
        # compute validation NDCG
        mae_list = []
        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, N_vad)
            X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')
        
            pred_val = sess.run(logits_var, feed_dict={dae.input_ph: X} )
            true_val = vad_data_te[idxlist_vad[st_idx:end_idx]]
            if sparse.isspmatrix(true_val):
                true_val = true_val.toarray()
            true_val = true_val.astype('float32')
            pred_val = pred_val[true_val > 0.0]
            true_val = true_val[true_val > 0.0]
            # print(pred_val, true_val)
            # input()
            pred_val = np.maximum(pred_val, 1.0)
            pred_val = np.minimum(pred_val, 5.0)
            mae = metrics.mean_absolute_error(true_val, pred_val)
            mse = metrics.mean_squared_error(true_val, pred_val)
            mae_list.append(mae)
        
        mae_ = np.asarray(mae_list).mean()
        print('epoch=%d mae=%.4f' % (epoch, mae_))
        maes_vad.append(mae_)

        # update the best model (if necessary)
        if mae_ > best_mae:
            saver.save(sess, '{}/model'.format(chkpt_dir))
            best_mae = mae_
