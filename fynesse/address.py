import numpy as np
import statsmodels.api as sm
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
from . import access


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


large_sum = ""
for column in access.column_list:
    large_sum += f"large_radius_counts.{column} + "
large_sum = large_sum[:-3]
small_sum = ""
for column in access.column_list:
    small_sum += f"small_radius_counts.{column} + "
small_sum = small_sum[:-3]

student_sql_query = f"""
SELECT
polygon_counts.oa21cd AS oa21cd,
polygon_counts.historic AS exact_historic,
large_radius_counts.historic AS large_historic,
small_radius_counts.university AS small_university,
polygon_counts.library AS exact_library,
small_radius_counts.library AS small_library,
large_radius_counts.fuel AS large_fuel,
small_radius_counts.convenience AS small_convenience,
small_radius_counts.bar AS small_bar,
polygon_counts.college AS exact_college,
large_radius_counts.synagogue AS large_synagogue,
large_radius_counts.leisure AS large_leisure,
ts062_data.L15 / ts062_data.Total_all_usual_residents AS student_percentage,
{large_sum} AS large_sum
FROM `polygon_counts`
JOIN `small_radius_counts` ON `polygon_counts`.`oa21cd` = `small_radius_counts`.`oa21cd`
JOIN `large_radius_counts` ON `polygon_counts`.`oa21cd` = `large_radius_counts`.`oa21cd`
JOIN `ts062_data` ON `ts062_data`.`geography_code` = `polygon_counts`.`oa21cd`
"""


def process_student_data(student_data):
    student_df = pd.DataFrame(student_data)
    student_df["large_sum"][student_df["large_sum"] == 0] = 1
    student_df.set_index("oa21cd", inplace=True)
    student_design_matrix = student_df[
        ["exact_library", "small_library", "small_convenience", "small_bar"]
    ]
    student_design_matrix["large_historic_normalised"] = (
        student_df["large_historic"] / student_df["large_sum"]
    )
    student_design_matrix["large_synagogue_normalised"] = (
        student_df["large_synagogue"] / student_df["large_sum"]
    )
    student_design_matrix["large_leisure_normalised"] = (
        student_df["large_leisure"] / student_df["large_sum"]
    )
    student_design_matrix["large_fuel_normalised"] = (
        student_df["large_fuel"] / student_df["large_sum"]
    )
    student_design_matrix["log_college_exact"] = np.log10(
        student_df["exact_college"] + 1
    )
    student_design_matrix["log_small_university"] = np.log10(
        student_df["small_university"] + 1
    )
    student_design_matrix["log_historic_exact"] = np.log10(
        student_df["exact_historic"] + 1
    )
    student_design_matrix.set_index(student_df.index, inplace=True)
    student_design_matrix = sm.add_constant(student_design_matrix)
    student_design_matrix = student_design_matrix.astype(float)
    return student_design_matrix, student_df["student_percentage"].astype(float)


def fit_validate_and_predict(
    design_matrix, response_vector, response_vector_name="Response Vector"
):
    model = sm.OLS(response_vector, design_matrix)
    results = model.fit()
    print(results.summary())
    cross_validation_score = cross_validate(
        design_matrix=design_matrix, response_vector=response_vector, k=10
    )
    print(f"Cross Validation Score: {cross_validation_score}")
    fig, ax = plt.subplots()
    ax.scatter(response_vector, results.fittedvalues, alpha=0.5)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"True vs. Predicted Values for {response_vector_name}")
    return lambda oa: results.fittedvalues[oa]
