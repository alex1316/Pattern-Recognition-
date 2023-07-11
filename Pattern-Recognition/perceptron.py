import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, num_iterations=50):
        self.learning_rate = learning_rate # 2: Choose P0
        self.num_iterations = num_iterations
        self.weights = None

    def fit(self, X, y):

        self.weights = np.random.rand(X.shape[1])
        # 1: Choose w randomly
        t = 0  # 3: Initialize t = 0

        while True: # 4: Repeat

            for j in range(X.shape[0]): # 4.2: For i=1 to N=sample in the data
                y_ = np.dot(X[j], self.weights)
                if y[j] * y_ <= 0:
                    # if misclasified
                    self.weights += self.learning_rate * y[j] * X[j] # 4.3

            self.learning_rate = self.learning_rate * 0.95 # 4.4: Adjust Pt
            t += 1 # 4.5: Increase t:t=t+1

            if t >= self.num_iterations:
                # Stop if maximum number of iterations is reached
                break


    def predict(self, X):
        return np.where(np.dot(X, self.weights) > 0, 1, -1)
