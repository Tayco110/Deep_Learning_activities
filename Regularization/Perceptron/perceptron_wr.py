import numpy as np

class PerceptronWithRegularization:
    def __init__(self, learning_rate=0.1, n_epochs=100, alpha=0.01):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha = alpha

    def fit(self, X, y):
        self.errors = []
        self.weights = np.zeros(1 + X.shape[1])

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += (update * xi) - (self.alpha * self.weights[1:])
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)