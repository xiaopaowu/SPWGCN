from __future__ import division
from __future__ import print_function
from turtle import pen
import numpy as np
import time
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings("ignore")
from utils import *
from models import GCN
from metrics import calculate_f_score
import os
import sys



def get_inverse(seq):
    ans = []
    for i in seq:
        if i == 0.:
            temp = 0.
        else:
            temp = 1. / i
        ans.append(temp)
    ans = np.array(ans)
    return ans

if len(sys.argv) != 4:
	sys.exit("Use: python train.py <dataset>")


pro = sys.argv[1]
ratio = sys.argv[2]
dataset =pro + "_my_method_"+ratio

seed = 48
tf.set_random_seed(seed)
# Setting
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.04, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay',0,
                'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                    'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
image_flag = int(sys.argv[3])
# image_flag = 1
adj, features, y_train, y_test, train_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset,image_flag)

y_all = y_train + y_test
all_weight = np.sum(y_all, axis=0)
all_sum = np.sum(all_weight, axis=0)
weight_sum = 1. / all_sum
weight = all_weight*weight_sum

all_max = np.max(all_weight)
all_weight = get_inverse(all_weight)
all_weight = all_weight * all_max
features = sp.identity(features.shape[0])

# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32),
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True, all_weight = all_weight)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
        
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

last_list = [str(0) for i in range(21)]
# Train model
t1 = time.time()
for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step                          
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                    model.layers[0].embedding], feed_dict=feed_dict)
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #     "train_acc=", "{:.5f}".format(
    #         outs[2]), "time=", "{:.5f}".format(time.time() - t))

    # Testing
    test_cost, test_acc, pred, labels, test_duration = evaluate(
        features, support, y_test, test_mask, placeholders)
    # print(pred)

    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(labels[i])

    path1 = 'results/' + dataset + '.txt'
    weight_las = calculate_f_score(test_labels, test_pred, 5,weight)

    # 选出加权f值最高的一次epoch
    if weight_las[20] > last_list[20]:
        last_list = weight_las

f = open(path1, 'a+')
a = last_list[0] + '\t' + last_list[1] + '\t' + last_list[2] + '\n' + last_list[3] + '\t' + last_list[4] + '\t' + last_list[5] + '\n' \
    + last_list[6] + '\t' + last_list[7] + '\t' + last_list[8] + '\n' + last_list[9] + '\t' + last_list[10] + '\t' + last_list[11] + '\n'  \
    + last_list[12] + '\t' + last_list[13] + '\t' + last_list[14] + '\n' + last_list[15] + '\t' + last_list[16] + '\t' + last_list[17] + '\n' \
    + last_list[18] + '\t' + last_list[19] + '\t' + last_list[20] + '\n' + "*********" +dataset+'\n'
f.write(a)
f.close()
