# This deep neural network model is contains only fully connected layers. It does not contains convolutional layers.
# This deep network is trained on kaggle data for facial expression recognition.

# tmp/data/1 is 32, 64, 512, 512, 2e-4 ==> 84%, 36%
# tmp/data/2 is 32, 64, 512, 512, 4e-4 ==> no improvement at all.
# tmp/data/3 is 32, 64, ksize = [1, 3, 3, 1], 1152, 576, 2e-4 with relu at opt layer ==> not imporving al all
# tmp/data/4 is 32, 64, 1152, 576, 2e-4 with relu at opt layer ==> no improvement at all.
# tmp/data/5 is 32, 64 ksize = [1, 3, 3, 1], 1152, 576, 3e-4, relu no applied at opt layer ==> 
import tensorflow as tf
import numpy as np
import csv


width = 48
height = 48
no_classes = 7
batch_size = 64
flatten_size = width * height
log_dir = './tmp/data/5'
tf.reset_default_graph()


file_handle = open("C:/Users/Devilal/Documents/Machine Learning/Data/FacialExpressionsKaggle/fer2013.csv", 'r')
reader = csv.reader(file_handle, delimiter=',')

def next_batch():
	cnt = 0
	labels = []
	images = []
	for row in reader:
		tag = row[0]
		idx = int(tag)
		pixels = row[1]
		label = [0, 0, 0, 0, 0, 0, 0]
		label[idx] = 1
		pixels = pixels.split(" ")
		image = np.array(pixels, dtype = np.float32)
		images.append(image)
		labels.append(label)
		cnt += 1
		if(cnt == batch_size):
			break
	return images, labels

def set_file_handle_to_beginning():
	file_handle.seek(0)
	next(reader)
	return


def preprocessing():
	cnt_train = 0
	cnt_test = 0
	set_file_handle_to_beginning()
	for row in reader:
		if(row[2] == 'Training'):
			cnt_train += 1
		else:
			cnt_test += 1
	return cnt_train, cnt_test

train_size, test_size = preprocessing();
set_file_handle_to_beginning()

def set_file_handle_to_test():
	for row in reader:
		if(row[1] == 'Training'):
			continue
		else:
			break

with tf.name_scope("input"):
	x = tf.placeholder(tf.float32, shape=[None, flatten_size], name='input_image')
	y = tf.placeholder(tf.int32, shape=[None, no_classes], name='input_label')

with tf.name_scope('total_cost_plac'):
	total_cost_plac = tf.placeholder(tf.float32, shape=())
with tf.name_scope('train_plac'):
	train_plac = tf.placeholder(tf.int32, shape=())
with tf.name_scope('test_plac'):
	test_plac = tf.placeholder(tf.int32, shape=())

def neural_network_model(input_x):

	#Reshaped to 4d tensor
	with tf.name_scope('Reshaped_to_4d_tensor'):
		reshaped = tf.reshape(input_x, [-1, width, height, 1])

	#parameters at convolutional layer 1
	with tf.name_scope('Params_at_conv_layer1'):
		weight_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
		bias_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))

	#convolutional layer 1
	with tf.name_scope('conv_layer1'):
		output_conv1 = tf.nn.relu(tf.nn.conv2d(reshaped, weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
	with tf.name_scope('maxpool_at_layer1'):
		output_maxpool1 = tf.nn.max_pool(output_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#parameters at convolutional layer 2
	with tf.name_scope("params_at_conv_layer2"):
		weight_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
		bias_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

	#convolutional layer 2
	with tf.name_scope('conv_layer2'):
		output_conv2 = tf.nn.relu(tf.nn.conv2d(output_maxpool1, weight_conv2, strides = [1, 1, 1, 1], padding='SAME') + bias_conv2)
	
	with tf.name_scope('maxpool_at_layer2'):
		output_maxpool2 = tf.nn.max_pool(output_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#flattening
	with tf.name_scope('flattened'):
		flatten_conv = tf.reshape(output_maxpool2, [-1, 12 * 12 * 64])

	#parameters at fullyconnected layer1, no. of neurons = flatten_size
	with tf.name_scope('params_at_layer_fc1'):
		weight_fc1 = tf.Variable(tf.truncated_normal(shape=[12* 12 * 64, 1152], dtype=tf.float32))
		bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1152], dtype=tf.float32))

	#Processing at fc1
	with tf.name_scope('Fc1'):
		output_fc1 = tf.add(tf.matmul(flatten_conv, weight_fc1), bias_fc1)
		activated_output_fc1 = tf.nn.relu(output_fc1)

	#parametes at fullyconnected layer2,
	with tf.name_scope('params_at_layer_fc2'):
		weight_fc2 = tf.Variable(tf.truncated_normal(shape=[1152, 576], dtype=tf.float32))
		bias_fc2 = tf.Variable(tf.constant(0.1, shape=[576], dtype=tf.float32))

	with tf.name_scope('FC2'):
		output_fc2 = tf.add(tf.matmul(activated_output_fc1, weight_fc2), bias_fc2);
		activated_output_fc2 = tf.nn.relu(output_fc2)

	#parameters at output layer
	with tf.name_scope('params_at_output_layer'):
		weight_output_layer = tf.Variable(tf.truncated_normal(shape=[576, no_classes], dtype=tf.float32))
		bias_output_layer = tf.Variable(tf.constant(0.1, shape=[no_classes], dtype=tf.float32))

	#result at output layer
	with tf.name_scope('output_layer'):
		opt_layer = tf.add(tf.matmul(activated_output_fc2, weight_output_layer), bias_output_layer)
		#activated_opt_layer = tf.nn.relu(opt_layer);
	return opt_layer

def build_and_train_model(x):
	pred_res = neural_network_model(x)
	with tf.name_scope("cross_entropy"):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_res, labels = y))
	with tf.name_scope('optimizer'):
		optimizer = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
	with tf.name_scope('truth_table_bool'):
		truth_table_bool = tf.equal(tf.argmax(pred_res, 1), tf.argmax(y, 1));
	with tf.name_scope('truth_table_int'):
		truth_table_int = tf.cast(truth_table_bool, tf.int32)
	with tf.name_scope('correct_preds'):
		correct_preds = tf.reduce_sum(truth_table_int)

	with tf.name_scope('merging_all'):
		tf.summary.scalar('cost', total_cost_plac)
		tf.summary.scalar('Train Accuracy', train_plac)
		tf.summary.scalar('Test Accuracy', test_plac)
		merge_op = tf.summary.merge_all()

	epochs = 100
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(log_dir)
		writer.add_graph(sess.graph)
		for ep in range(epochs):

			set_file_handle_to_beginning()
			#Training
			total_cost = 0
			for i in range(int(train_size/batch_size)):
				x_input, y_input = next_batch()
				_, cost= sess.run([optimizer, cross_entropy], feed_dict = {x : x_input, y : y_input})
				total_cost += cost

			#evaluating Training data
			set_file_handle_to_beginning()
			train_correct_preds = 0
			train_evaluations = 0
			for i in range(int(train_size/batch_size)):
				x_input, y_input = next_batch()
				train_correct_preds += sess.run(correct_preds, feed_dict={x : x_input, y : y_input})
				train_evaluations += batch_size
			#print("epoch : ", ep, " cost: ", total_cost, " train Acc : ", total_correct_preds, "/", total_evaluations, end='')
			
			#Testing
			set_file_handle_to_test()
			test_correct_preds = 0
			test_evaluations = 0
			for i in range(int(test_size/batch_size)):
				x_input, y_input = next_batch()
				test_correct_preds += sess.run(correct_preds, feed_dict={x : x_input, y : y_input})
				test_evaluations += batch_size

			merged_summary = sess.run(merge_op, feed_dict={total_cost_plac:total_cost, train_plac:train_correct_preds, test_plac:test_correct_preds})
			writer.add_summary(merged_summary, ep + 1)

			print("epoch : ", ep + 1, " cost: ", total_cost, "  train acc: ", train_correct_preds, "/", train_evaluations, " test acc: ", test_correct_preds, "/", test_evaluations)
		
		writer.close()

build_and_train_model(x)
file_handle.close()
