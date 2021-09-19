import tensorflow as tf
from DataCleaning import DataCleaning

# Defining main function 
def main():
    dataCleaning = DataCleaning()
    dataCleaning.clean(4)
    #dataCleaning.generateSentenceEmbedding()
    dataCleaning.generateWord2VecEmbedding()
    dataCleaning.save()

    

# Using the special variable  
# __name__ 
if __name__=="__main__":
    print(tf.__version__)
    main()