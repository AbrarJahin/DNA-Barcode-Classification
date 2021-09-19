import tensorflow as tf
from DataCleaning import DataCleaning

# Defining main function 
def main():
    dataCleaning = DataCleaning()
    dataCleaning.upDownScale()
    dataCleaning.clean()
    dataCleaning.preprocess(4)
    #dataCleaning.generateSentenceEmbedding()
    #dataCleaning.generateWord2VecEmbedding()
    dataCleaning.generateDoc2VecEmbedding(epochs=100)
    dataCleaning.save()


# Using the special variable  
# __name__ 
if __name__=="__main__":
    print(tf.__version__)
    main()