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
		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		self.model_filename = "../model/" + model_filename
		#Define The Model
		self.model = tf.keras.Sequential([
			tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(1, activation='sigmoid')
		])
		# Compile the model
		self.model.compile(
				loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy', self.precision_m, self.recall_m]
			)
		print(model.summary())

	def trainAndSaveModel(self):
		# Fit the RNN - Training
		history = self.model.fit(self.X_tr, self.y_tr['labels'], 
                    batch_size=32,
					epochs=10,
                    validation_data=(self.X_test, self.y_test))
		print(history)
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