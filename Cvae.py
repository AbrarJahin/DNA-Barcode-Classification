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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Cvae(object):
	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500, model_filename = 'Cvae.sav', output_dimention=30):
		embedding_column_count = number_of_trees

		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		self.model_filename = "../model/" + model_filename

		#Define The Model
		
		original_inputs = tf.keras.Input(shape=(self.X_tr.shape[1],1), name="encoder_input")
		x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
		z_mean = layers.Dense(output_dimention, name="z_mean")(x)
		z_log_var = layers.Dense(output_dimention, name="z_log_var")(x)
		z = Sampling()((z_mean, z_log_var))
		self.encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

		latent_inputs = tf.keras.Input(shape=(output_dimention,), name="z_sampling")
		x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
		outputs = layers.Dense(original_dim, activation="sigmoid")(x)
		self.decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

		outputs = decoder(z)
		self.vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

		#Define logger
		self.logger = keras.callbacks.TensorBoard(
			log_dir='logs',
			write_graph=True,
			histogram_freq=5
		)

		optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
		self.vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
		
		print(self.vae.summary())

	def trainAndSaveModel(self):
		history = self.vae.fit(self.X_tr, self.X_tr, epochs=30, batch_size=128)
		print(history)
		#Save the model
		pickle.dump(self.vae, open(Utils.getAbsFilePath(self.model_filename), 'wb'))	#Store Model to File
		return

	def restoreModel(self):
		self.vae = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
		result = self.vae.score(self.X_test, self.X_test)
		print(result)
		print("RNN-Model Loaded Successfully")

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		y_pred = self.model.predict(X_pred)
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df