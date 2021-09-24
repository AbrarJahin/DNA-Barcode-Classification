import keras.backend as K
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, MaxPool1D, BatchNormalization, Dropout, Input
from keras.models import Sequential
from Utils import Utils
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow as tf
import pandas as pd
import datetime
import math
import numpy as np
import keras
#from resnet import Residual
import math
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import sklearn.metrics as metrics

class Cnn(object):
	#Lambda Functions - Start
	def recall_m(self, y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision_m(self, y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	#Lambda Functions - End

	def __init__(self, X_tr, y_tr, X_test, y_test, model_filename = 'Cnn1D.sav', epochs = 10, batch_size = 512):
		self.epochs = epochs
		self.batch_size = batch_size

		block_1_layers, block2_layers, block3_layers = 32, 16, 8
		int(math.sqrt(32))

		#https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f
		# For CNN, RNN and HAN

		self.y_tr = to_categorical(y_tr['labels'].values, dtype = "uint8")
		self.y_test = to_categorical(y_test['labels'].values, dtype = "uint8")

		self.X_tr = X_tr.values.reshape(X_tr.values.shape[0], X_tr.values.shape[1], 1)
		self.X_test = X_test.values.reshape(X_test.values.shape[0], X_test.values.shape[1], 1)

		self.model_filename = "../model/" + model_filename

		#self.y_tr.shape[1] = 1214

		#Define The Model
		model = tf.keras.Sequential()
		#model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', input_shape=(self.X_tr.shape[1],1), name='block1_conv1'))
		model.add(Input(shape=(self.X_tr.shape[1],1), batch_size=None, name="Input Layer"))
		################################################################################
		model.add(Conv1D(filters=block_1_layers, kernel_size=5, strides=1, activation='relu', name='block1_conv1'))
		model.add(MaxPool1D(pool_size=int(math.sqrt(block_1_layers)), name='block1_pool1'))
		#model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))
		model.add(Dense(block_1_layers, activation='relu', name='block1_dense1'))
		model.add(Dropout(0.1, name='block1_drop1'))
		model.add(BatchNormalization(momentum=0.6, epsilon=1e-5, axis=1, name='block1_bn1'))

		model.add(Conv1D(filters=block2_layers, kernel_size=3, strides=1, activation='relu', name='block2_conv1'))
		model.add(MaxPool1D(pool_size=int(math.sqrt(block2_layers)), name='block2_pool1'))
		model.add(Dense(block2_layers, activation='relu', name='block2_dense1'))
		#model.add(Flatten(name='block1_flat1'))
		model.add(Dropout(0.1, name='block2_drop1'))
		model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, axis=1, name='block2_bn1'))

		model.add(Conv1D(filters=block3_layers, kernel_size=3, strides=1, activation='relu', name='block3_conv1'))
		model.add(MaxPool1D(pool_size=int(math.sqrt(block3_layers)), name='block3_pool1'))
		model.add(Dense(block3_layers, activation='relu', name='block3_dense1'))
		#model.add(MaxoutDense(512, nb_feature=4, name="block2_maxout2"))
		model.add(Dropout(0.1, name='block3_drop1'))

		#model.add(Dense(16, activation='relu', name='block2_dense3', input_dim=5,
		#	kernel_initializer='ones',
		#	kernel_regularizer=tf.keras.regularizers.L1(0.01),
		#	activity_regularizer=tf.keras.regularizers.L2(0.01)))

		################################################################################
		#model.add(Reshape((None, self.y_tr.shape[1]), name='block4_reshape1'))
		model.add(Flatten(name='predict_flatten1'))
		model.add(Dense(
					self.y_tr.shape[1],
					activation='softmax',
					name="predict"
				))
		self.model = model

		#Define logger
		self.logger = keras.callbacks.TensorBoard(
			log_dir='logs',
			write_graph=True,
			histogram_freq=2
		)
		# Compile the model
		self.model.compile(
				loss='categorical_crossentropy',
				#optimizer="adam",
				optimizer=tf.keras.optimizers.Adam(learning_rate=.00005, beta_1=0.901, beta_2=0.9995, epsilon=1e-08),
				#optimizer='rmsprop',
				#optimizer=tf.keras.optimizers.Adam(
				#		learning_rate=0.1,
				#		beta_1=0.9,
				#		beta_2=0.999,
				#		epsilon=1e-07,
				#		amsgrad=False,
				#	),
				#optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.005, decay=0.0004),
				metrics=['accuracy', self.precision_m, self.recall_m]
				#metrics=[metrics]
			)
		for i in range(1,len(model.layers)):
			print(model.layers[i-1].output_shape, model.layers[i].input_shape, model.layers[i-1].output_shape == model.layers[i].input_shape)
		print(self.model.summary())

	def trainAndSaveModel(self):
		# self.X_tr.shape => (14834, 784)
		# self.y_tr.shape => (14834, 1)
		# unique label - 1202 (by set) / 1204(by categorical)
		# Fit the RNN - Training
		history = self.model.fit(
						self.X_tr,
						self.y_tr, 
						batch_size = self.batch_size,
						epochs = self.epochs,
						validation_data = (self.X_test, self.y_test),
						shuffle = True,
						verbose = 2,
						callbacks = [self.logger]
					)
		print(history)
		test_error_rate = self.model.evaluate(self.X_test, self.y_test, verbose=1)
		print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
		#Save the model
		try:
			pickle.dump(self.model, open(Utils.getAbsFilePath(self.model_filename), 'wb'))	#Store Model to File
		except Exception as e:
			print("CNN-Model Save Failed")
			print(e)
		return

	def restoreModel(self):
		try:
			self.model = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
			result = self.model.score(self.X_test, self.y_test)
			print(result)
			print("CNN-Model Loaded Successfully")
		except Exception as e:
			print("CNN-Model Loaded Failed")
			print(e)

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		#x_pred_reshaped = X_pred.values.reshape(X_pred.values.shape[0], self.dimention, self.dimention, 1)
		x_pred_reshaped = X_pred.values.reshape(X_pred.values.shape[0], X_pred.values.shape[1], 1)
		y_pred = self.model.predict(x_pred_reshaped)
		y_pred = list(map(np.argmax, y_pred))
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df