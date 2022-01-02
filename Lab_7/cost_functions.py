import numpy as np
import math

class GaussianCostFunction:
    """
    Implements a cost function for fitting a Gaussian (normal) distribution.
    """
    def __init__(self, features, y_true):
        """
        The constructor takes the feature matrix and true y values
        for the training data.
        """
        self.features = features
        self.y_true = y_true
        
    
    def predict(self, features, params):
        """
        Predicts the y values for each data point
        using the feature matrix and the model parameters.
        
        We expect that the features are a Nx1 matrix of
        x values.  The params is a length-2 array of the
        mean (mu) and std deviation (sigma).
        """
        mu = params[0]
        sigma = params[1]
        left_side = 1 / (sigma * (math.sqrt(2 * math.pi)))
        right_side = math.e ** (-.5 * ((features - mu) / sigma) ** 2)
        return left_side * right_side

    def _mse(self, y_true, pred_y):
        """
        Calculates the mean-squared error between the predicted and
        true y values.
        """
        return (1/len(y_true))*sum((y_true - pred_y)**2)
        
        
    def cost(self, params):
        """
        Calculates the cost function value for the model's predictions
        using the given params.
        
        This should:
        1. Use the params and data's features to predict the y values
        2. Calculate the error between the true and predicted y values
        3. Return the error
        """
        pred_y = self.predict(self.features, params)
        return self._mse(self.y_true, pred_y)
        
class LinearCostFunction:
    """
    Implements a cost function for a linear regression model.
    """
    def __init__(self, features, y_true):
        """
        The constructor takes the feature matrix and true y values
        for the training data.
        
        """
        self.features = features
        self.y_true = y_true
        
    
    def predict(self, features, params):
        """
        Predicts the y values for each data point
        using the feature matrix and the model parameters.
        
        We expect that the features are a NxM matrix.
        The params are a 1D array of length M.
        """
        return np.dot(features, params)
        
    def _mse(self, y_true, pred_y):
        """
        Calculates the mean-squared error between the predicted and
        true y values.
        """
        return (1 / len(y_true)) * sum((y_true - pred_y) ** 2)
        
    def cost(self, params):
        """
        Calculates the cost function value for the model's predictions
        using the given params.
        
        This should:
        1. Use the params and data's features to predict the y values
        2. Calculate the error between the true and predicted y values
        3. Return the error
        """
        pred_y = self.predict(self.features, params)
        return self._mse(self.y_true, pred_y)
