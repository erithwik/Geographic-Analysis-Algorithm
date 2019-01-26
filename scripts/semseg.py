import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import random
import math

import os
import sys
from multiprocessing.pool import ThreadPool


sys.path.append('E:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/bin')

float_t = tf.float32
np_float_t = np.float32

trainingIters = 45000
_batch_size = 200
display_step, display_iters = 150, 20 # a display step is batch size x iters feed forwards, to get more accurate data
imshow_step = 3000
save_step = 3000
dropout = 0.7
reg_param = 0.0004 # regularization parameter
decision_boundary = 0.4 # (anything above is class 1, anything below is class 0)
learning_rate = 4e-7
learning_rate_decay = (1/4500) # learning rate is calculated as LR_0 / (1 + step * decay)

_imSize = 48
_oSize = 16

logs_path = '/tmp/tensorflow_logs/example'


def load_imset(path, ext, label):
	ret = []
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			if os.path.splitext(os.path.join(root, name))[1].lower() == ext:
				im = Image.open(os.path.join(root, name))
				ret.append(np.uint8(np.asarray(im,dtype=np_float_t)))
				#ret.append(im)
	return ret
	

def load_images():
	input_path = os.getcwd() + "/../semseg_data/input"
	label_path = os.getcwd() + "/../semseg_data/label"
	all_inputs = load_imset(input_path, ".png", False)
	all_labels = load_imset(label_path, ".png", True)

	return all_inputs, all_labels

all_inputs, all_labels = load_images()

#returns numpy array for input and label images
def getSegmentFromImage(arrayIndex, test):

	while True:

		pad = math.ceil(int(_imSize)*math.sqrt(2) + 8)
		startX = random.randint(4, 1500-pad)
		startY = random.randint(4, 1500-pad)
		
		pad2 = math.ceil(int(_imSize)*math.sqrt(2)/2)*2
		endX = startX + pad2
		endY = startY + pad2

		angle = random.uniform(0,360)

		
		_loc_Start = (pad2 - _imSize) / 2
		_loc_End = _loc_Start + _imSize
		
		inputImage = Image.fromarray(all_inputs[arrayIndex][startY:endY, startX:endX])
		if(not test):
		        inputImage = inputImage.rotate(angle, resample=Image.LINEAR, expand=False)
		inputImage = inputImage.crop((_loc_Start,_loc_Start,_loc_End,_loc_End))

		oLoc_Start = (pad2 - _oSize) / 2
		oLoc_End = oLoc_Start + _oSize

		outputImage = Image.fromarray(all_labels[arrayIndex][startY:endY, startX:endX])
		if(not test):
		        outputImage = outputImage.rotate(angle, resample=Image.LINEAR, expand=False)
		outputImage = outputImage.crop((oLoc_Start,oLoc_Start,oLoc_End,oLoc_End))


		inputImage = np.array(inputImage)
		outputImage = np.array(outputImage)
		
		if(np.all(inputImage[0,0] > 240) or np.all(inputImage[_imSize-1,0] > 246) or np.all(inputImage[0,_imSize-1] > 246) or np.all(inputImage[_imSize-1,_imSize-1] > 246) ):
			continue

		inputImage = np.reshape(inputImage,(-1))
		outputImage = np.reshape(outputImage[:,:,0], (-1))

		if( np.size(inputImage) == _imSize * _imSize * 3 and np.size(outputImage) == _oSize * _oSize):
			return inputImage, outputImage

		
	'''
	while True:
	
		pad = int(_imSize - _oSize) + 8
		startX = random.randint(4, 1500-pad)
		startY = random.randint(4, 1500-pad)
		endX = startX + _imSize
		endY = startY + _imSize
		
		inputImage = all_inputs[arrayIndex][startX:endX, startY:endY]

		if( np.size(inputImage) != _imSize * _imSize * 3 ):
			continue

		if(np.all(inputImage[0,0] > 240) or np.all(inputImage[_imSize-1,0] > 246) or np.all(inputImage[0,_imSize-1] > 246) or np.all(inputImage[_imSize-1,_imSize-1] > 246) ):
			continue

		inputImage = np.reshape(inputImage,(-1))


		border = int((_imSize - _oSize)/2)
		startX_L = startX + border
		startY_L = startY + border
		endX_L = endX - border
		endY_L = endY - border
		
		labelImage = np.reshape(all_labels[arrayIndex][startX_L:endX_L, startY_L:endY_L, 0],(-1))

		if( np.size(inputImage) == _imSize * _imSize * 3 and np.size(labelImage) == _oSize * _oSize):
			return inputImage, labelImage
	'''

def make_batch(size_test):
	size, test = size_test

	input_array = []
	label_array = []

	for i in range(size):
		indx = 0
		if(test):
			indx = random.randint(0,int(0.2 * len(all_inputs)))
		else:
			indx = random.randint(int(0.2 * len(all_inputs)),len(all_inputs)-1)
		inputI, labelI = getSegmentFromImage(indx,test)
		input_array.append(inputI)
		label_array.append(labelI)

	#convert from list to numpy array, then reshape
	input_batch = np.concatenate(input_array)
	label_batch = np.concatenate(label_array)
	input_batch = np.reshape(input_batch,[-1,_imSize*_imSize*3])
	label_batch = np.reshape(label_batch,[-1,_oSize*_oSize*1])

	return input_batch, label_batch





#dataset_mean = tf.Variable(tf.truncated_normal([None, _imSize * _imSize * 3], stddev = 0.1))
#dataset_std = tf.Variable(tf.truncated_normal([None, _imSize * _imSize * 3], stddev = 0.1))





x_in = tf.placeholder(float_t,[None, _imSize * _imSize * 3]) 
y_in = tf.placeholder(float_t,[None, _oSize * _oSize * 1])
prob = tf.placeholder(float_t)
lr = tf.placeholder(float_t)
step = tf.placeholder(tf.int32)

x = tf.reshape(x_in,[-1,_imSize*_imSize*3])


#x = tf.image.per_image_standardization(x)
#x = tf.reshape(x,[-1,_imSize * _imSize * 3])
y = tf.divide(y_in, 255) #convert color space from [0,255] to [0,1]

weightdist = 0.005
conv_weights = 0.01

weights = {
	'w1': tf.Variable(tf.truncated_normal([9,9,3,64], stddev = conv_weights, dtype=float_t), name = "weight_1"),
	'w2': tf.Variable(tf.truncated_normal([7,7,64,128], stddev = conv_weights, dtype=float_t), name = "weight_2"),
	'w3': tf.Variable(tf.truncated_normal([5,5,128,128], stddev = conv_weights, dtype=float_t), name = "weight_3"),
	'wd1': tf.Variable(tf.truncated_normal([_imSize*_imSize*32, 4096], stddev = weightdist, dtype=float_t), name = "full_weight_1"),
	'out': tf.Variable(tf.truncated_normal([4096, _oSize * _oSize], stddev = weightdist, dtype=float_t), name = "full_weight_2") #check
}

biases = {
	'b1': tf.Variable(tf.truncated_normal([64], stddev = weightdist, dtype=float_t), name = "bias_1"),
	'b2': tf.Variable(tf.truncated_normal([128], stddev = weightdist, dtype=float_t), name = "bias_2"),
	'b3': tf.Variable(tf.truncated_normal([128], stddev = weightdist, dtype=float_t), name = "bias_3"),
	'bd1': tf.Variable(tf.truncated_normal([4096], stddev = weightdist, dtype=float_t), name = "full_bias_1"),
	'out': tf.Variable(tf.truncated_normal([_oSize * _oSize], stddev = weightdist, dtype=float_t), name = "full_bias_2")
}

def conv2d(x, w, b, strides=1):
	x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1], padding = 'SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def fullyConnected(x, w, b):
	fc = tf.add(tf.matmul(x, w), b)
	return fc

def maxPool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def convNet(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1, _imSize, _imSize, 3])
	conv1 = conv2d(x, weights['w1'], biases['b1'])
	conv1 = maxPool2d(conv1, k=2)
	conv2 = conv2d(conv1, weights['w2'], biases['b2'])
	conv3 = conv2d(conv2, weights['w3'], biases['b3'])
	#conv3 = maxPool2d(conv3, k=2)

	fc1 = tf.reshape(conv3, [-1,weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)

	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	out = tf.sigmoid(out) * 0.9999 + 0.00005
	
	return out

pred = convNet(x, weights, biases, prob)

regularization_each = [tf.nn.l2_loss(w) for w in [ weights['w1'], weights['w2'], weights['wd1'], weights['out'], weights['w3'] ] ]
total_reg = tf.add_n(regularization_each)
total_reg = total_reg * ( reg_param / _batch_size )

cost = tf.reduce_mean( -y*tf.log(pred)-(1-y)*tf.log(1-pred) ) + total_reg
#cost = tf.reduce_mean( tf.square(pred-y) )
optimizer = tf.train.AdamOptimizer(2e-5).minimize(cost)

prediction_image = tf.ceil(pred-decision_boundary)

accuracy = tf.metrics.accuracy(y,prediction_image)
precision = tf.metrics.precision(y,prediction_image)
recall = tf.metrics.recall(y,prediction_image)
f1_score = (2.0 * tf.cast(precision,tf.float32) * tf.cast(recall,tf.float32)) / (tf.cast(precision,tf.float32) + tf.cast(recall,tf.float32))


init = tf.global_variables_initializer()
initL = tf.local_variables_initializer()

saver = tf.train.Saver()

num_threads = 4
pool = ThreadPool(processes=num_threads) # create pool for multithreading making batches

with tf.Session() as sess:
	
	sess.run(init)
	sess.run(initL)

	saver.restore(sess, "../semseg_data/save.ckpt") # save is the place that we start from to make modifications
	
	#all_data = np.array(all_inputs)
	#mean = np.mean(all_data, axis=(0,1,2))
	#var = np.std(all_data, axis=(0,1,2))

	acc_over_time = []
	f1_over_time = []

	batch_x, batch_y = make_batch((_batch_size,False)) # initial batch is generated single-thread

	step = 1
	
	while step < trainingIters:

		alpha = learning_rate / (1 + learning_rate_decay * step)

		#initiate process for multithreading
		async_result = pool.map_async(make_batch, [( int( round(_batch_size/4) ) ,False) for i in range(4)] ) 


		_,tC = sess.run([optimizer, cost], feed_dict={x_in:batch_x, y_in:batch_y, prob:dropout, lr:alpha})

		if step%display_step == 0:

			#sess.reset([accuracy,precision,recall,f1_score])
			
			sess.run(initL)
			a,f1,c,cT,p = 0,0,0,0,0
			for i in range(display_iters):
				batch_xd, batch_yd = make_batch((_batch_size,True))
				c,a,p,r,f1 = sess.run([cost,accuracy, precision, recall, f1_score], feed_dict={x_in:batch_xd, y_in:batch_yd, prob:1.0, lr:alpha})
				cT += c
			cT /= display_iters

			
			print("") #that's right padding goes before, what are you gonna, do sue me?
			print("Iteration: " + str(step))
			print("Train -- loss: " + str(tC))
			print("Test  -- loss: " + str(cT) + " acc: " + str(a[1]) + " f1: " + str(f1[1]) + " prec: " + str(p[1]))

			acc_over_time.append(a[1])
			f1_over_time.append(f1[1])
			
			
			'''print out an image every once in a while, if you're 

			if(step%imshow_step == 0):
				prediction, pred_image = sess.run([pred, prediction_image], feed_dict={x_in:batch_xd, y_in:batch_yd, prob:1.0, std:var, _mean:mean, lr:alpha})
				in_image = Image.fromarray(np.reshape(np.uint8(batch_xd), [_imSize*_batch_size_T,_imSize,3]),'RGB')
				out_image = Image.fromarray(np.reshape(prediction, [_oSize*_batch_size,_oSize])*255)
				y_image = Image.fromarray(np.reshape(batch_yd, [_oSize*_batch_size,_oSize]))
				p_image = Image.fromarray(np.reshape(pred_image, [_oSize*_batch_size,_oSize])*255)
				
				in_image.show()
				out_image.show()
				y_image.show()
				p_image.show()

			 '''
				

		if(step%save_step == 0):
			save_path = saver.save(sess, "../semseg_data/save2.ckpt")
			print("Model saved in file: %s", str(save_path))
		step += 1
		
		batches = async_result.get()

		batches_x = []
		batches_y = []

		for i in range(num_threads):
			single_batch = batches[i]
			single_batch_x, single_batch_y = single_batch
			batches_x.append(single_batch_x)
			batches_y.append(single_batch_y)
			
		batch_x = np.concatenate(batches_x)
		batch_y = np.concatenate(batches_y)
			

	statfile = open('training_goodies.txt', 'w')
	statfile.write(str(acc_over_time) + "\n")
	statfile.write(str(f1_over_time))
	

	


