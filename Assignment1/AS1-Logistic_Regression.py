################################### Problem 1.1 ###################################

def learn_mul(X, y):
    ################# YOUR CODE COMES HERE ######################
    # training and return the multi-class logistic model

    #############################################################
    return lr

def inference_mul(x, lr_model):
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values

    #############################################################
    return y_pred


################################### Problem 1.2 ###################################


def learn_mul2bin(X, y):
    lrs = []
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    for i in range(num_classes):
        print('training %s classifier'%(ordinal(i+1)))
        ################# YOUR CODE COMES HERE ######################
        # training and return the multi-class logistic model

        #############################################################

    return lrs

def inference_mul2bin(X, lrs):
    ################# YOUR CODE COMES HERE ######################
    # inference model and return predicted y values

    #############################################################
    return y_pred


################################### Problem 2   ###################################


class LogisticRegression:
    def __init__(self, ): # <<< You can add your own input parameters
        ################# YOUR CODE COMES HERE ######################
        # initialize class member variable

        #############################################################

    def sigmoid(self, z):
        # YOUR CODE COMES HERE
        return

    def fit(self, X, y):
        ################# YOUR CODE COMES HERE ######################
        # training model here

        #############################################################
        return

    def predict(self, X):
        ################# YOUR CODE COMES HERE ######################
        # return predicted y

        #############################################################
        return

    # You can add your own member functions