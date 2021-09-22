import tensorflow as tf
import configparser
from DataCleaning import DataCleaning
from RandomForest import RandomForest
from Rnn import Rnn
from FFNet import FFNet
from Cnn import Cnn
from Cnn2D import Cnn2D
from BiLstm import BiLstm
from Svm import Svm

# Defining main function 
def main():
    try:
        config = configparser.ConfigParser()
        config.read('.env')
        #Embedding Config
        isEmbiddingDone = config.get('Default','isEmbiddingDone') == 'True'
        embedding = config.get('Default','embedding') #"sbert", "w2vec" "d2vec"
        perWordLength = int(config.get('Default','perWordLength'))
        outputColumnCount = int(config.get('Default','outputColumnCount')) #Embedding row number
        wordsWindowSize = int(config.get('Default','wordsWindowSize'))
        epochCount = int(config.get('Default','epochCount'))
        minIgnoreCount = int(config.get('Default','minIgnoreCount'))
        isTrainingDone = config.get('Default','isTrainingDone') == 'True'
        trainingModel = config.get('Default','trainingModel')
        batchSize = int(config.get('Default','batchSize'))
        ifUpscaleNeeded = config.get('Default','ifUpscaleNeeded') == 'True'
    except Exception as e:
        print(e, "=> Default valus set from code")
        isEmbiddingDone = False
        embedding = "onehot"
        perWordLength = 4
        outputColumnCount = 784 #28*28 for 2D CNN
        wordsWindowSize = 50
        epochCount = 1000
        minIgnoreCount = 2
        isTrainingDone = False
        trainingModel = "CNN" # "RNN", "CNN", ......
        batchSize = 512
        ifUpscaleNeeded = True
    #Start Embedding
    dataCleaning = DataCleaning()
    if not isEmbiddingDone:
        if ifUpscaleNeeded: dataCleaning.upDownScale()
        dataCleaning.clean()
        dataCleaning.preprocess(perWordLength)
        if embedding=="sbert":
            dataCleaning.generateSentenceEmbedding()
        elif embedding=="w2vec":
            dataCleaning.generateWord2VecEmbedding(vector_size=outputColumnCount, window=wordsWindowSize, epochs=epochCount, min_count=minIgnoreCount)
        elif embedding=="d2vec":
            dataCleaning.generateDoc2VecEmbedding(vector_size=outputColumnCount, window=wordsWindowSize, epochs=epochCount, min_count=minIgnoreCount)
        elif embedding == "onehot":
            dataCleaning.generateOneHotEncoding(vector_size=outputColumnCount, window=wordsWindowSize, epochs=epochCount, min_count=minIgnoreCount)
        dataCleaning.save()

    (X_tr,y_tr), (X_test,y_test) = dataCleaning.getTrainTestSplit(embedding = embedding)
    X_pred = dataCleaning.getXTest(embedding = embedding)
    totalNoOfLebels = len(dataCleaning.lebels)  #Not Needed

    #Resnet - need to implement, CNN need to be implemented
    model = None
    if trainingModel == "RndomForest": #Random Forest
        model = RandomForest(X_tr, y_tr, X_test, y_test, totalNoOfLebels)
        #if not isTrainingDone:
        #    model.trainAndSaveModel()
        #model.restoreModel()
        #model.savePrediction(X_pred)
    elif trainingModel == "CNN": #CNN
        model = Cnn(X_tr, y_tr, X_test, y_test, epochs = epochCount, batch_size = batchSize)
    elif trainingModel == "CNN2D": #CNN
        model = Cnn2D(X_tr, y_tr, X_test, y_test, epochs = epochCount, batch_size = batchSize)
    elif trainingModel == "SVM": #SVM
        model = Svm(X_tr, y_tr, X_test, y_test)
    elif trainingModel == "LSTM": #LSTM
        model = Lstm(X_tr, y_tr, X_test, y_test, epochs = epochCount, batch_size = batchSize)
    elif trainingModel == "BiLSTM": #BiLSTM
        model = BiLstm(X_tr, y_tr, X_test, y_test, epochs = epochCount, batch_size = batchSize)
    elif trainingModel == "FFNet": #FFNet
        model = FFNet(X_tr, y_tr, X_test, y_test, totalNoOfLebels)
    elif trainingModel == "RNN": #RNN
        model = Rnn(X_tr, y_tr, X_test, y_test, totalNoOfLebels)

    if not isTrainingDone:
        model.trainAndSaveModel()
    else:
        model.restoreModel()
    model.savePrediction(X_pred)

# Using the special variable  
# __name__ 
if __name__=="__main__":
    print(tf.__version__)
    main()