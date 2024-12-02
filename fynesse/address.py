# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

def cross_validate(design_matrix, response_vector, k, n = 10, regularised = False, alpha = None, L1_wt = 1):
  test_scores = []
  if alpha == None:
    n, p = design_matrix.shape
    alpha = 1.1 * np.sqrt(n) * sps.norm.ppf(1 - 0.05 / (2 * p))
  for _ in range(n):
    permutation = np.random.permutation(len(design_matrix))
    design_matrix_folds = [design_matrix.iloc[permutation[i::k]] for i in range(k)]  
    response_vector_folds = [response_vector.iloc[permutation[i::k]] for i in range(k)]  
    for i in range(k):
      train_design_matrix = np.concatenate(design_matrix_folds[:i] + design_matrix_folds[i+1:])
      train_response_vector = np.concatenate(response_vector_folds[:i] + response_vector_folds[i+1:])
      model = sm.OLS(train_response_vector, train_design_matrix*1)
      fit = 0
      if regularised:
        fit = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
      else:
        fit = model.fit()
      predictions = fit.predict(train_design_matrix)
      rsum = np.sum((predictions - train_response_vector)**2)
      tss = np.sum((train_response_vector - np.mean(train_response_vector))**2)
      rsquared = 1 - rsum/tss
      test_scores.append(rsquared)
  return sum(test_scores)/len(test_scores)
