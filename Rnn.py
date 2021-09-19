
class Rnn(object):
	#Lambda Functions - Start
	def cleanString(self, text: str)-> str:
		#remove punctuation
		text = "".join([char for char in text if char not in string.punctuation])
		# Convert all to lower
			#not needed in here
		# Remove Stop Words
			#not needed in here
		return text

	def splitWords(self, dna_seq: str, word_len: int)-> str:
		strArray = [dna_seq[index : index + word_len] for index in range(0, len(dna_seq), word_len)]
		text = " ".join(strArray)
		return text
	#Lambda Functions - End

	def __init__(self, train_x_file = "train_features.csv", train_y_file = "train_labels.csv"):
		dataX = pd.read_csv(self.getAbsFilePath("data/" + train_x_file))
		dataY = pd.read_csv(self.getAbsFilePath("data/" + train_y_file))
		self.total_data = dataX
		self.total_data['labels'] = dataY['labels']
