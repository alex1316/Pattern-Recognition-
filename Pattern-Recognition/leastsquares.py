# Making imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Least_Squares():
    def __int__(self):
        self.m = 0
        self.c = 0

    def train(self, X,Y):
        # Building the model
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y)

        num = 0
        den = 0
        for i in range(len(X)):
            #num += (X[i] - X_mean) * (Y[i] - Y_mean)
            #den += (X[i] - X_mean) ** 2
            num += np.matmul((X[i] - X_mean).reshape(-1, 1), (Y[i] - Y_mean).reshape(1, 1))
            den += np.matmul((X[i] - X_mean).reshape(-1, 1), (X[i] - X_mean).reshape(1, -1))
        self.m = num / den
        #self.c = Y_mean - self.m * X_mean
        self.c = Y_mean - np.dot(self.m, X_mean)

        #print(self.m, self.c)

    def predict(self, X):
        # Making predictions
        #Y_pred = self.m * X + self.c
        Y_pred = np.dot(X, self.m) + self.c
        return Y_pred

"""
plt.scatter(X, Y)  # actual
# plt.scatter(X, Y_pred, color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  #predicted
plt.show()"""