# Facial-Expression-Recognition using Tensorflow
	
Facial Emotion recognition is very easy task for human, as we have a very complex and sophisticated biological neural network in our brain which has been trained since we born. But it is very difficult task for computer machines. Here I provide a neural network implementation to perform facial expression recognition. It implements a simple but efficient convolution neural network using most popular library tensorflow.

# Prerequisites
* Tensorflow version latest by 1.1, see how to [install](https://www.tensorflow.org/install/)
* Csv lib
* Knowledge of deep learning concepts, if you don't feel comfortable working with cnn then you can use [online book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/index.html).
* Facial expression data set must be available on your system, [download here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

# Data-sets
The available data sets contains 7 basic emotions: happy, sad, disgust, surprise, fear, anger and neutral. It comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each labeled with one of the 7 emotion classes. This tells that our cnn model outputs either probabilities or class score into 7 classes. I used 28672 number of images for training our neural network model and 7168 number of images for testing purpose.

# The Model
	It uses csv python module to open given csv file into appropriate csv module. Here we use 5 layers.
		1. Convolutional layer 
			Input  : 4d tensor, dim:[N, w, h, Number of input channel = 1], where N is batch size.
			Output : 4d tensor, dim:[N, w/2, h/2, Number of filters at cnn layer-1]

		2. Convolutional layer 
			Input  : 4d tensor, dim:[N, w/2, h/2, Number of filters at cnn layer-1]
			Output : 4d tensor, dim:[N, w/4, h/4, Number of filters at cnn layer-2]

			Now this output 4d tensor is flattened inorder to provide input to fully connected layer-1.

		3. Fully connected layer
			Input  : 2d tensor, dim:[N, Flattened size]
			Output : 2d tenser, dim:[N, Number of neurons at fully connected layer-1]

		4. Fully connected layer
			Input  : 2d tensor, dim:[N, Number of neurons at fully connected layer-1]
			Output : 2d tenser, dim:[N, Number of neurons at fully connected layer-2]

		5. Output layer.
			Input  : 2d tensor, dim:[N, number of neurons at fully connected layer-2]
			Output : 2d tenser, dim:[N, Number of classes]

# How to run
Simply run python file.

# Model graph
![graph goes here](https://github.com/Devbishnoi29/Facial-Expression-Recognition/blob/master/images/tfgraph.png)

# Plot between Cost and Epochs
![cost plot](https://github.com/Devbishnoi29/Facial-Expression-Recognition/blob/master/images/Cost.PNG)

# Plot between Training Accuracy and Epochs
![Train accuracy](https://github.com/Devbishnoi29/Facial-Expression-Recognition/blob/master/images/TrainAcc.PNG)

# Plot between Testing Accuracy and Epochs
![Test Accuracy](https://github.com/Devbishnoi29/Facial-Expression-Recognition/blob/master/images/TestAcc.PNG)

# About me
I am a computer programmer who loves to solve programming problems and exploring the exciting possibilities using deep learning. I am interested in solving real life problems using efficient algorithms and computer vision that creates innovative solutions to real-world problems. I hold a B.Tech degree in computer Engineering From Nit kurukshetra. You can reach me on [LinkedIn](https://www.linkedin.com/in/devi-lal-468596126/).