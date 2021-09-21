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

	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500, model_filename = 'Rnn.sav'):
		embedding_column_count = number_of_trees

		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		self.model_filename = "../model/" + model_filename
		#Define The Model
		self.model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(
					filters=32,
					kernel_size=(3, 3),
					activation='relu',
					input_shape=(28, 28, 3),
					padding='same'
				),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
			tf.keras.layers.MaxPooling2D((2, 2)),
			tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
			tf.keras.layers.MaxPooling2D((2, 2)),
			tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10, activation='softmax')
		])
		#Define logger
		self.logger = keras.callbacks.TensorBoard(
			log_dir='logs',
			write_graph=True,
			histogram_freq=5
		)
		# Compile the model
		self.model.compile(
				loss='categorical_crossentropy',#binary_crossentropy
				optimizer=tf.keras.optimizers.Adam(
						learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
						name='Adam Optimizer'
					),
				#optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.005, decay=0.0004),
				metrics=['accuracy', self.precision_m, self.recall_m]
			)
		print(self.model.summary())

	def trainAndSaveModel(self):
		# self.X_tr.shape => (14834, 784)
		# self.y_tr.shape => (14834, 1)
		# Fit the RNN - Training
		history = self.model.fit(
						self.X_tr,
						self.y_tr['labels'], 
						batch_size=32,
						epochs=10,
						validation_data=(self.X_test, self.y_test),
						shuffle=True,
						verbose=2,
						callbacks=[self.logger]
					)
		print(history)
		test_error_rate = self.model.evaluate(self.X_test, self.y_test, verbose=1)
		print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
		#Save the model
		pickle.dump(self.model, open(Utils.getAbsFilePath(self.model_filename), 'wb'))	#Store Model to File
		return

	def restoreModel(self):
		self.model = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
		result = self.model.score(self.X_test, self.y_test)
		print(result)
		print("RNN-Model Loaded Successfully")

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		y_pred = self.model.predict(X_pred)
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df