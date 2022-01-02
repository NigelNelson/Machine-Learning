import numpy as np
import scipy
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """
    def __init__(self, k, aggregation_function):
        """
        Takes two parameters.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. The
        aggregation_function is either "mode" for classification or
        "average" for regression.
        
        Parameters
        ----------
        k : int
           Number of neighbors
        
        aggregation_function : {"mode", "average"}
           "mode" : for classification
           "average" : for regression.
        """
        self.k = k
        if aggregation_function == "mode" or "average":
            self.aggregation_function = aggregation_function
        else:
            raise ValueError("Invalid Aggregation Function Specified"
                             ", must be \'mode\' or \'average\'")
       
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        
        Parameters
        ----------
        X : 2D-array of shape (n_samples, n_features) 
            Training/Reference data.
        y : 1D-array of shape (n_samples,) 
            Target values.
        """

        self.X = X
        self.y = y
        
        
    def predict(self, X):
        """
        Predicts the output variable's values for the query points X.
        
        Parameters
        ----------
        X : 2D-array of shape (n_queries, n_features)
            Test samples.
            
        Returns
        -------
        y : 1D-array of shape (n_queries,) 
            Class labels for each query.
        """
        distances = spatial.distance.cdist(X, self.X)
        sorted_distances = np.argsort(distances, axis=1)[:, :self.k]
        labels = self.y[sorted_distances]
        if self.aggregation_function == 'mode':
            return scipy.stats.mode(labels, axis=1).mode.flatten()
        else:
            return np.mean(labels, 1)
