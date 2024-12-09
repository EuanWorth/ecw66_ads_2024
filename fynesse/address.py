import numpy as np
import statsmodels.api as sm
import scipy.stats as sps


def cross_validate(
    design_matrix, response_vector, k, n=10, regularised=False, alpha=None, L1_wt=1
):
    test_scores = []
    if alpha is None and regularised:
        count, p = design_matrix.shape
        alpha = 1.1 * np.sqrt(count) * sps.norm.ppf(1 - 0.05 / (2 * p))
    for j in range(n):
        permutation = np.random.permutation(len(design_matrix))
        design_matrix_folds = [design_matrix.iloc[permutation[i::k]] for i in range(k)]
        response_vector_folds = [
            response_vector.iloc[permutation[i::k]] for i in range(k)
        ]
        for i in range(k):
            train_design_matrix = np.concatenate(
                design_matrix_folds[:i] + design_matrix_folds[i + 1 :]
            )
            train_response_vector = np.concatenate(
                response_vector_folds[:i] + response_vector_folds[i + 1 :]
            )
            test_design_matrix = design_matrix_folds[i]
            test_response_vector = response_vector_folds[i]
            model = sm.OLS(train_response_vector, train_design_matrix * 1)
            fit = 0
            if regularised:
                fit = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
            else:
                fit = model.fit()
            predictions = fit.predict(test_design_matrix)
            rsum = np.sum((predictions - test_response_vector) ** 2)
            tss = np.sum((test_response_vector - np.mean(test_response_vector)) ** 2)
            rsquared = 1 - rsum / tss
            test_scores.append(rsquared)
    return sum(test_scores) / len(test_scores)
