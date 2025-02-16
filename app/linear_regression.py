# linear_regression.py

import numpy as np

# Base Linear Regression
class LinearRegression:
    def __init__(self, regularization, lr, method, momentum, xavier_method, mlflow_params, num_epochs=100, batch_size=50):
        self.regularization = regularization
        self.lr = lr
        self.method = method
        self.momentum = momentum
        self.xavier_method = xavier_method
        self.mlflow_params = mlflow_params
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # ✅ Add 'theta' attribute if missing
        if not hasattr(self, 'theta'):
            self.theta = np.zeros((5, 1))  # Default shape (adjust if needed)

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model is not trained. 'theta' is missing.")
        return X @ self.theta


# Regularization Classes
class LassoPenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta): 
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta): 
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class NoRegularization:
    def __init__(self, l=None): 
        pass

    def __call__(self, theta): 
        return 0

    def derivation(self, theta):
        return 0


# Lasso Model
class Lasso(LinearRegression):
    def __init__(self, method, lr, xavier_method, momentum, l, mlflow_params, num_epochs=100, batch_size=50):
        regularization = LassoPenalty(l)
        super().__init__(regularization, lr, method, momentum, xavier_method, mlflow_params, num_epochs, batch_size)
        # ✅ Fix missing theta on loaded models
        if not hasattr(self, 'theta'):
            self.theta = np.zeros((5, 1))


# Ridge Model
class Ridge(LinearRegression):
    def __init__(self, method, lr, xavier_method, momentum, l, mlflow_params, num_epochs=100, batch_size=50):
        regularization = RidgePenalty(l)
        super().__init__(regularization, lr, method, momentum, xavier_method, mlflow_params, num_epochs, batch_size)
        if not hasattr(self, 'theta'):
            self.theta = np.zeros((5, 1))


# Normal Model
class Normal(LinearRegression):
    def __init__(self, method, lr, xavier_method, momentum, l=None, mlflow_params=None, num_epochs=100, batch_size=50):
        regularization = NoRegularization(l)
        super().__init__(regularization, lr, method, momentum, xavier_method, mlflow_params, num_epochs, batch_size)
        if not hasattr(self, 'theta'):
            self.theta = np.zeros((5, 1))
