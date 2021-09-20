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

class Rnn(object):
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
			#tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
			tf.keras.layers.Embedding(
					input_dim = self.X_tr.shape[0],
					output_dim = embedding_column_count,
					input_length=self.X_tr.shape[1]
				),
			#tf.keras.layers.Dropout(0.1),
			tf.keras.layers.Bidirectional(
					tf.keras.layers.LSTM(
							128,
							return_sequences=True
						)
				),
			tf.keras.layers.Bidirectional(
					tf.keras.layers.LSTM(64)
				),
			tf.keras.layers.Dense(
					32,
					activation='relu'
				),
			tf.keras.layers.Dense(
					1,
					activation='sigmoid'
				)
		])
		self.logger = keras.callbacks.TensorBoard(
			log_dir='logs',
			write_graph=True,
			histogram_freq=5
		)
		# Compile the model
		self.model.compile(
				loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy', self.precision_m, self.recall_m]
			)
		print(model.summary())

	def trainAndSaveModel(self):
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