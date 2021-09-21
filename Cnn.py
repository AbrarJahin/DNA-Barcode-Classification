import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
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

	def __init__(self, X_tr, y_tr, X_test, y_test, model_filename = 'Rnn.sav', epochs = 10, batch_size = 512):
		self.epochs = epochs
		self.batch_size = batch_size
		self.dimention = int(math.sqrt(X_tr.shape[1]))
		self.X_tr = X_tr.values.reshape(X_tr.values.shape[0], self.dimention, self.dimention, 1)
		self.y_tr = to_categorical(y_tr['labels'].values, dtype = "uint8")
		self.X_test = X_test.values.reshape(X_test.values.shape[0], self.dimention, self.dimention, 1)
		self.y_test = to_categorical(y_test['labels'].values, dtype = "uint8")

		self.model_filename = "../model/" + model_filename

		#Define The Model
		self.model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(
					filters=32,
					kernel_size=(3, 3),
					activation='relu',
					input_shape=(self.dimention, self.dimention, 1),
					padding='same'
				),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Conv2D(filters=math.ceil(self.y_tr.shape[1]*1.1), kernel_size=(3, 3), activation='relu', padding='valid', name='conv_1'),
			tf.keras.layers.MaxPooling2D((2, 2)),
			tf.keras.layers.Conv2D(filters=math.ceil(self.y_tr.shape[1]/8), kernel_size=(3, 3), activation='relu'),
			tf.keras.layers.MaxPooling2D((2, 2)),
			tf.keras.layers.Conv2D(filters=math.ceil(self.y_tr.shape[1]/64), kernel_size=(3, 3), activation='relu'),
			tf.keras.layers.MaxPooling2D((2, 2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(self.y_tr.shape[1], activation='softmax')
		])
		#Define logger
		self.logger = keras.callbacks.TensorBoard(
			log_dir='logs',
			write_graph=True,
			histogram_freq=5
		)
		# Compile the model
		self.model.compile(
				loss='categorical_crossentropy',
				optimizer="adam",
				#optimizer=tf.keras.optimizers.Adam(
				#		learning_rate=0.1,
				#		beta_1=0.9,
				#		beta_2=0.999,
				#		epsilon=1e-07,
				#		amsgrad=False,
				#	),
				#optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.005, decay=0.0004),
				metrics=['accuracy', self.precision_m, self.recall_m]
			)
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
			print("RNN-Model Save Failed")
			print(e)
		return

	def restoreModel(self):
		try:
			self.model = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
			result = self.model.score(self.X_test, self.y_test)
			print(result)
			print("RNN-Model Loaded Successfully")
		except Exception as e:
			print("RNN-Model Loaded Failed")
			print(e)

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		x_pred_reshaped = self.X_pred.values.reshape(X_pred.values.shape[0], self.dimention, self.dimention, 1)
		y_pred = self.model.predict(x_pred_reshaped)
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df