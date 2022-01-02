import numpy as np


class Optimizer:
    """
    Implements Gradient Descent using numerical differentiation for calculating the gradient.
    """
    def __init__(self, step_size, max_iter, tol, delta):
        """
        Max_iter -- maximum number of iterations to run
        step_size -- also known as lambda
        tol --
        delta -- perturbation to use in numerical differentiation
        """
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.delta = delta
        
    
    def optimize(self, cost_func, starting_params):
        """
        Finds parameters that optimize the given cost function.
        
        This method should implement your iterative algorithm for updating your parameter estimates.
        Use an updated estimate of the gradient to update the parametes.
        
        Give consideration for what the exit conditions of this loop should be.
        
        Returns a tuple of (optimized_param, iters)
        """
        iters = 0
        is_tol_met = False
        new_params = params = starting_params
        while iters < self.max_iter and not is_tol_met:
            params = new_params
            gradient = self._gradient(cost_func, params)
            new_params = self._update(params, gradient)
            dif = self._calculate_change(params, new_params)
            is_tol_met = dif < self.tol
            iters += 1
        return params, iters


    def _calculate_change(self, old, new):
        """
        Calculates the change between the old and new parameters.
        Returns a scalar.
        """
        return np.linalg.norm(new - old)


    def _gradient(self, cost_func, params):
        """
        Numerically estimates the gradient (first derivative) of the cost function
        at param.
        
        First-order numerical differentiation
        df/dx = [ f(x + delta) - f(x) ] / delta
        
        Should return the gradient at the caluclated point
        """
        gradient = np.zeros(params.size)
        for i in range(params.size):
            partial = np.copy(params)
            partial[i] += self.delta
            gradient[i] = (cost_func.cost(partial) - cost_func.cost(params)) / self.delta
        return gradient
        
            
    def _update(self, param, gradient):
        """
        Updates the param vector using the Gradient Descent algorithm.                
        
        Returns the new parameters.  (Do not modify input)
        """
        return param - gradient * self.step_size
