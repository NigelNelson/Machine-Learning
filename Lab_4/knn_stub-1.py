import numpy as np
import scipy as sci
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
        self.aggregation_function = aggregation_function
       
        
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
        self.labeled_data = np.hstack(X/np.linalg.norm(X), y)
        
        
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

        concatenated_data = np.hstack(self.labeled_data, X/np.linalg.norm(X))
        indexed_data = np.hstack(concatenated_data, np.arange(0, concatenated_data.shape[0]))
        distanced_data = np.hstack(indexed_data, scipy.spatial.cdist(self.X, X))


   