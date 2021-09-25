import pandas as pd
import numpy as np
import os
import time
import requests
import ast

#File Locations
rawData = "./data/1632442103_submission.csv"
processedData = "./data/incidents_processed.csv"

rawDataFrame = pd.read_csv(rawData)#, converters={'labels_old':lambda x: np.array(ast.literal_eval(x))}
rawDataFrame["labels"] = 0
for index, row in rawDataFrame.iterrows():
	try:
		result = rawDataFrame.at[index, 'labels_old']
		#.apply(lambda x: np.array(x)
		#result = rawDataFrame.at[index, 'labels_old'].apply(lambda x: 
  #                         np.fromstring(
  #                             x.replace('\n','')
  #                              .replace('[','')
  #                              .replace(']','')
  #                              .replace('  ',' '), sep=' '))
		rawDataFrame.at[index, 'labels'] = 1
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
rawDataFrame.to_csv(processedData)