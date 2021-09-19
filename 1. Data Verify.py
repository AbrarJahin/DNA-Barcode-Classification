import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer

train_features = pd.read_csv('./data/train_features.csv')
train_labels = pd.read_csv('./data/train_labels.csv')
test_features = pd.read_csv('./data/test_features.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')
#train_features
print(train_features.keys(), train_labels.keys())
total_train_data = train_features
total_train_data['labels'] = train_labels['labels']

total_train_data.to_csv('./data/total_train_data.csv')

"""**PreProcess Data**"""
row_to_remove_index, label_index = [], defaultdict(list)
unique = set()
for index, row in total_train_data.iterrows():
	try:
		total_train_data.at[index, 'dna'] = total_train_data.at[index, 'dna'].replace('-', '')
		if len(total_train_data.at[index, 'dna']) == 0: row_to_remove_index.append(index)
		unique.add(total_train_data.at[index, 'labels'])
		label_index[total_train_data.at[index, 'labels']].append(index)
	except Exception as err:
		print(f'Error occurred during updating row of train_dna: {err}')

total_train_data.drop(total_train_data.index[row_to_remove_index], inplace=True)

total_train_data
uniqueList = sorted(list(unique))
print(uniqueList[0], uniqueList[-1], len(uniqueList))
print(label_index[1])

"""**Analyze the data Format - Input Type**"""

for k in label_index:
		if len(label_index[k])<4:
			print(k)
			print(label_index[k])

"""Create Sample DF"""

#This section is for trying data len
res_df = train_features.append(test_features, ignore_index=True)
print(len(res_df),len(train_features),len(test_features))
#res_df

"""Embedding Library"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings(action='once')

#!pip install sentence-transformers
# %pip install  --upgrade sentence-transformers


embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#embedding = embedding_model.encode(res_df.at[index, 'dna'])

"""**Save Embedding**"""

for index, row in train_features.iterrows():
  try:
	embedding = embedding_model.encode(train_features.at[index, 'dna'].replace('-', ''))
	for embadeIndex, val in enumerate(embedding):
	  train_features.at[index, "paraphrase-MiniLM-L6-v2_embedding_" + str(embadeIndex).zfill(3)] = val
  except Exception as err:
	print(f'Error occurred during File Save: {err}')
train_features

for index, row in test_features.iterrows():
  try:
	embedding = embedding_model.encode(test_features.at[index, 'dna'].replace('-', ''))
	for embadeIndex, val in enumerate(embedding):
	  test_features.at[index, "paraphrase-MiniLM-L6-v2_embedding_" + str(embadeIndex).zfill(3)] = val
  except Exception as err:
	print(f'Error occurred during File Save: {err}')
test_features.head()

import tensorflow as tf
tf.test.gpu_device_name()

"""**Train Set Defination**"""

y_tr =  train_labels[['labels']].values
X_tr = train_features[[s for s in train_features.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]].values

"""**Preprocessing Data**"""

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.0,1.0))  #Keras needs the data to be fitted between 0-1
X_tr = scaler.fit_transform(X_tr)
X_tr

"""**Define the model**"""

from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Dense(500, input_dim=384, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")

"""**Train the model**"""

model.fit(
	X_tr,
	y_tr,
	epochs=50,
	shuffle=True,
	verbose=2
)

test_error_rate = model.evaluate(X_tr, y_tr, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

prediction = model.predict(X_tr)

# Grab just the first element of the first prediction (since that's the only have one)
prediction[0][0], y_tr[0][0]