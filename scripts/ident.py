import tensorflow as tf
from PIL import Image
import numpy as np
import time
import random

import os
import sys

dropout = 0.5

trainingIters = 45000
im_size = 256
num_classes = 2
batch_size = 50
display_step, display_iters = 150, 15 # a display step is batch size x iters feed forwards, to get more accurate data
save_step = 3000

reg_param = 0.2 # regularization parameter
learning_rate = 4e-5
learning_rate_decay = (1/4500) # learning rate is calculated as LR_0 / (1 + step * decay)

num_occupied = 8000
num_vacant = 6000
num_examples = num_occupied+num_vacant

# load all images in a directory as a list of numpy arrays
def load_imset(path, ext):
	ret = []
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			if os.path.splitext(os.path.join(root, name))[1].lower() == ext:
				im = Image.open(os.path.join(root, name))
				im = im.resize((im_size,im_size), Image.LANCZOS)
				ret.append(np.asarray(im))
	return ret
	
# create training set from files
def load_images():
	occ_path = os.getcwd() + "/../identification_tset/occupied_processed" # all parcels containing buildings
	vac_path = os.getcwd() + "/../identification_tset/vacant_processed"   # all parcels containing no buildings

	# the list of buildings that are occupied + their labels
	occ_images = load_imset(occ_path, ".png")
	occ_labels = [np.array([0,1]) for i in range(num_occupied)] # TODO: replace with text file containing commercial/residential/etc. labels

	vac_images = load_imset(vac_path, ".png")
	vac_labels = [np.array([1,0]) for i in range(num_vacant)]   # all label 0, since 0 = vacant

	#combine both sets
	all_inputs = occ_images + vac_images
	all_labels = occ_labels + vac_labels

	#shuffle lists
	random_indices = random.sample(range(num_examples), num_examples)
	all_inputs = [all_inputs[i] for i in random_indices]
	all_labels = [all_labels[i] for i in random_indices]
	
	return all_inputs, all_labels


all_inputs, all_labels = load_images()

# get a random selection of images from the training set
def make_batch(size, test):
	inputs, labels = [], []

	# get random sample of length batch_size
	random_indices = 0
	if(test):
		random_indices = random.sample(range(int(0.2*num_examples)), size)
	else:
		random_indices = random.sample(range(int(0.2*num_examples), num_examples), size)
	inputs = [all_inputs[i] for i in random_indices]
	labels = [all_labels[i] for i in random_indices]

	# convert from list to numpy array
	input_batch = np.concatenate(inputs)
	label_batch = np.concatenate(labels)

	input_batch = np.reshape(input_batch, (-1,im_size*im_size*3))
	label_batch = np.reshape(label_batch, (-1, num_classes))
	
	return input_batch,label_batch

x_in = tf.placeholder(tf.float32, [None, im_size * im_size * 3])
y = tf.placeholder(tf.float32, [None, num_classes])
prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
step = tf.placeholder(tf.int32)

x = tf.reshape(x_in,[-1,im_size*im_size,3])

init_weight = 0.001

weights = {
	'w1': tf.Variable(tf.truncated_normal([12,12,3,32], stddev = init_weight)),
	'w2': tf.Variable(tf.truncated_normal([6,6,32,32], stddev = init_weight)),
	'wd1': tf.Variable(tf.truncated_normal([1024*8,1024], stddev = init_weight)),
	'wd2': tf.Variable(tf.truncated_normal([1024,1024], stddev = init_weight)),
	'out': tf.Variable(tf.truncated_normal([1024,num_classes], stddev = init_weight))
}

biases = {
	'b1': tf.Variable(tf.truncated_normal([32], stddev = init_weight)),
	'b2': tf.Variable(tf.truncated_normal([32], stddev = init_weight)),
	'bd1': tf.Variable(tf.truncated_normal([1024], stddev = init_weight)),
	'bd2': tf.Variable(tf.truncated_normal([1024], stddev = init_weight)),
	'out': tf.Variable(tf.truncated_normal([num_classes], stddev = init_weight))
}

def conv2d(x, w, b, strides = 1):
	x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1], padding = 'SAME')
	x = tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

def fullyConnected(x,w,b):
	fc = tf.add(tf.matmul(x,w),b)
	return fc

def maxPool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def convNet(x, weights, biases, drouput):
	x = tf.reshape(x, shape = [-1, im_size, im_size, 3])
        
	conv1 = conv2d(x, weights['w1'], biases['b1'], 2)
	pool1 = maxPool2d(conv1, k=2)
	conv2 = conv2d(pool1, weights['w2'], biases['b2'], 2)
	pool2 = maxPool2d(conv2, k=2)

	fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)

	fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
	fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
	fc2 = tf.nn.relu(fc2)
	fc2 = tf.nn.dropout(fc2, dropout)

	out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
	out = tf.nn.softmax(out)
	return out

pred = convNet(x, weights, biases, prob)

regularization_each = [tf.nn.l2_loss(w) for w in [ weights['w1'], weights['w2'], weights['wd1'], weights['out'], weights['wd2'] ] ]
total_reg = tf.add_n(regularization_each)
total_reg = total_reg * ( reg_param / batch_size )

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + total_reg

prediction = tf.argmax(pred,1)

accuracy = tf.metrics.accuracy(tf.argmax(y, 1),prediction)
precision = tf.metrics.precision(tf.argmax(y, 1),prediction)
recall = tf.metrics.recall(tf.argmax(y, 1),prediction)
f1_score = (2.0 * tf.cast(precision,tf.float32) * tf.cast(recall,tf.float32)) / (tf.cast(precision,tf.float32) + tf.cast(recall,tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()
initL = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

	sess.run(init)
	sess.run(initL)

	all_data = np.array(all_inputs)
	acc_over_time = []
	f1_over_time = []
	
	saver.restore(sess, "../save/not_semseg (94% accuracy)/semantic_segmentation.ckpt")
	
	step = 1
	while step < trainingIters:

		alpha = learning_rate / (1 + learning_rate_decay * step)

		batch_x, batch_y = make_batch(batch_size,False)
		_,tC = sess.run([train_step, cost], feed_dict={x_in:batch_x, y:batch_y, prob:dropout, lr:alpha})

		if step%display_step == 0:
			
			sess.run(initL)
			a_t,f1_t,c_t = 0,0,0
			for i in range(display_iters):
				sess.run(initL)
				batch_xd, batch_yd = make_batch(batch_size,True)
				c,a,p,r,f1 = sess.run([cost,accuracy, precision, recall, f1_score], feed_dict={x_in:batch_xd, y:batch_yd, prob:1.0, lr:alpha})
				a_t += a[1]
				f1_t += f1[1]
				c_t += c
			a = a_t   / display_iters
			f1 = f1_t / display_iters
			c = c_t  / display_iters
			print("") #that's right padding goes before, what are you gonna, do sue me?
			print("Iteration: " + str(step))
			print("Train -- loss: " + str(tC))
			print("Test  -- loss: " + str(c) + " acc: " + str(a) + " f1: " + str(f1))
			acc_over_time.append(a)
			f1_over_time.append(f1)
			
		step += 1

		if(step%save_step == 0):
			save_path = saver.save(sess, "../save/semantic_segmentation.ckpt")
			print("Model saved in file: %s", str(save_path))

	statfile = open('training_goodies_2.txt', 'w')
	statfile.write(str(acc_over_time) + "\n")
	statfile.write(str(f1_over_time) + "\n")


