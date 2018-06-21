import itertools
import numpy as np
import pandas as pd
from time import time
import cvxopt.solvers
import numpy.linalg as la
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):

    @staticmethod
    def linear():
        # Implementing the linear method relation between two features x and y
        return lambda x,y:np.dot(x.T,y)

    @staticmethod
    def polykernel(dimension, offset):
        return lambda x, y: ((offset + np.dot(x.T,y)) ** dimension)

    @staticmethod
    def radial_basis(gamma):
        return lambda x, y: np.exp(-gamma*la.norm(np.subtract(x, y)))

class SVMTrainer(object):

    def __init__(self, kernel, c):
        # Assigning the attributes kernal and C value
        self.kernel = kernel
        self.c = c

    def train(self, X, y):
        # Training function
        # Caluculate the langrange multipliers
        lagrange_multipliers = self.compute_multipliers(X, y)
        # Trainer returns SVM predictor that is used to predict which class the test element belongs to
        return self.construct_predictor(X, y, lagrange_multipliers)

    def kernel_matrix(self, X, n_samples):
        # Size of kernal matrix is (no_of_inputs , no_of_inputs)
        # Reason for this is that kernel function value is calculated between every 2 inputs given
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        #  Returns the kernel function values
        return K

    def construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self.kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def compute_multipliers(self, X, y):
        # n_samples is no_of_inputs
        # n_features is no_of_features
        n_samples, n_features = X.shape
        # Returns kernel function matrix
        K = self.kernel_matrix(X,n_samples)
        # np.outer(a,b) gives
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        # Create a diagonal matrix of (n_samples , n_samples) dimension with -1 as value
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(y, (1, n_samples) , 'd')
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # To flatten as one dimension array
        return np.ravel(solution['x'])

class SVMPredictor(object):

    # Initializing the SVM predictor
    # Needs kernel (K) , bias (b) , weights (w) , support_vectors and support_vector_labels
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        # Weights is equal to support vector labels as per mathematical formulation
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        # result = b
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            # result += w * support_vector_labels * K
            result += z_i * y_i * self._kernel(x_i, x)
        # Returning the sign of the value predicted
        # +1 means belonging to positive (non spam) class
        # -1 means belonging to negative (spam) class
        return np.sign(result).item()

def calculate(true_positive,false_positive,false_negative,true_negative):
    result = {}
    result['precision'] = true_positive / (true_positive + false_positive)
    result['recall'] = true_positive / (true_positive + false_negative)
    return result

def confusion_matrix(true_positive,false_positive,false_negative,true_negative):
    matrix = PrettyTable([' ', 'Ham' , 'Spam'])
    matrix.add_row(['Ham', true_positive , false_positive])
    matrix.add_row(['Spam', false_negative , true_negative])
    return matrix , calculate(true_positive,false_positive,false_negative,true_negative)

def implementSVM(X_train,Y_train,X_test,Y_test,parameters,type):
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0
    if(type=="polykernel"):
        dimension = parameters['dimension']
        offset = parameters['offset']
        trainer = SVMTrainer(Kernel.polykernel(dimension,offset),0.1)
        predictor = trainer.train(X_train,Y_train)
    elif(type=="linear"):
        trainer = SVMTrainer(Kernel.linear(),0.1)
        predictor = trainer.train(X_train,Y_train)
    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        if(ans==-1 and Y_test[i]==-1):
            spam_spam+=1
        elif(ans==1 and Y_test[i]==-1):
            spam_ham+=1
        elif(ans==1 and Y_test[i]==1):
            ham_ham+=1
        elif(ans==-1 and Y_test[i]==1):
            ham_spam+=1
    return confusion_matrix(ham_ham,ham_spam,spam_ham,spam_spam)

def write_to_file(matrix,result,parameters,type,start_time):
    f = open("results.txt","a")
    if(type=="polykernel"):
        f.write("Polykernel model parameters")
        f.write("\n")
        f.write("Dimension : " + str(parameters['dimension']))
        f.write("\n")
        f.write("Offset : " + str(parameters['offset']))
    elif(type=="linear"):
        f.write("Linear model")
    f.write("\n")
    f.write(matrix.get_string())
    f.write("\n")
    f.write("Precision : " + str(round(result['precision'],2)))
    f.write("\n")
    f.write("Recall : " + str(round(result['recall'],2)))
    f.write("\n")
    f.write("Time spent for model : " + str(round(time()-start_time,2)))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()

global_start_time = time()

cvxopt.solvers.options['show_progress'] = False

df1 = pd.read_csv('wordslist.csv')
df2 = pd.read_csv('frequency.csv',header=0)

input_output = df2.as_matrix(columns=None)
X = input_output[:,:-1]
Y = input_output[:,-1:]

total = X.shape[0]
train = int(X.shape[0] * 70 / 100)

X_train = X[:train,:]
Y_train = Y[:train,:]
X_test = X[train:,:]
Y_test = Y[train:,:]

f = open("results.txt","w+")
f.close()
k=0

type = {}
parameters = {}

type['1'] = "polykernel"
type['2'] = "linear"

for i in range(2,4):
    for j in range(0,10):
        start_time = time()
        parameters['dimension'] = i
        parameters['offset'] = j
        matrix , result = implementSVM(X_train,Y_train,X_test,Y_test,parameters,str(type['1']))
        write_to_file(matrix,result,parameters,type,start_time)
        k+=1
        print("Done : " + str(k))

start_time = time()
matrix , result = implementSVM(X_train,Y_train,X_test,Y_test,parameters,str(type['2']))
write_to_file(matrix,result,parameters,type,start_time)
k+=1
print("Done : " + str(k))

f = open("results.txt","a")
f.write("Time spent for entire code : " + str(round(time()-global_start_time,2)))
f.close()
