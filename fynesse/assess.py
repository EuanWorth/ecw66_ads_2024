from . import access
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
from sklearn.cluster import KMeans

size_table_name_map = {
    "exact": "polygon",
    "small": "small_radius",
    "large": "large_radius",
}
occupations_tables_map = {
    2001: "occupations_2001_data_rezoned",
    2021: "occupations_2021_data",
}


def display_heatmap(df, title, ax=None, use_rows=False):
    if ax is None:
        plt.matshow(df)
        if use_rows:
            plt.yticks(np.arange(len(df.index)), list(df.index), fontsize=14)
        else:
            plt.yticks(
                range(df.select_dtypes(["number"]).shape[1]),
                df.select_dtypes(["number"]).columns,
                fontsize=14,
            )
        plt.xticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
            rotation=90,
        )
        plt.title(title, fontsize=16)
        plt.colorbar()
        plt.show()
    else:
        im = ax.imshow(df)
        ax.set_xticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
            rotation=90,
        )
        if use_rows:
            ax.set_yticks(np.arange(len(df.index)), list(df.index), fontsize=14)
        else:
            ax.set_yticks(
                range(df.select_dtypes(["number"]).shape[1]),
                df.select_dtypes(["number"]).columns,
                fontsize=14,
            )
        ax.set_title(title, fontsize=16)
        plt.colorbar(im, ax=ax)


def display_price_graphs(gdf, place_name):
    _, ax = plt.subplots()
    ax.set_title(f"Prices in {place_name}")
    price = gdf["Price"].apply(lambda x: math.log(x))
    x = gdf.centroid.x
    y = gdf.centroid.y
    ax.scatter(x, y, c=price)
    plt.show()
    small_df = pd.DataFrame.from_dict({"price": price, "Longitude": x, "Latitude": y})
    display_heatmap(
        small_df.corr(),
        f"Correlation Matrix for Latitude and Longitude against prices in {place_name}",
    )


def do_analysis(latitude, longitude, place_name, conn):
    pp_gdf = access.gen_gdf(latitude, longitude, conn)
    osm_gdf = access.get_buildings(latitude, longitude)
    osm_gdf["Street"] = osm_gdf["addr:street"].apply(lambda x: str.upper(str(x)))
    join_df = pd.merge(
        pp_gdf,
        osm_gdf,
        left_on=["House Number", "Street"],
        right_on=["addr:housenumber", "Street"],
        how="inner",
    )
    join_gdf = join_df.set_geometry("geometry_y")
    display_price_graphs(join_gdf, place_name)


def display_osm_aggregate_summaries(conn):
    fig, axes = plt.subplots(nrows=len(access.feature_list), ncols=3, figsize=(15, 100))
    for feature, ax in zip(access.feature_list, axes):
        if isinstance(feature, tuple):
            feature_type, feature_name = feature
        else:
            feature_name = feature
            feature_type = None
        exact_data = access.sql_select(
            conn, f"SELECT COUNT(*) FROM polygon_counts GROUP BY {feature_name};"
        )
        small_data = access.sql_select(
            conn, f"SELECT COUNT(*) FROM small_radius_counts GROUP BY {feature_name};"
        )
        large_data = access.sql_select(
            conn, f"SELECT COUNT(*) FROM large_radius_counts GROUP BY {feature_name};"
        )
        exact_df = pd.DataFrame(exact_data, columns=["count"])
        small_df = pd.DataFrame(small_data, columns=["count"])
        large_df = pd.DataFrame(large_data, columns=["count"])
        ax[0].bar(exact_df.index, np.log10(exact_df["count"]))
        ax[1].bar(small_df.index, np.log10(small_df["count"]))
        ax[2].bar(large_df.index, np.log10(large_df["count"]))
        if feature_type == "shop":
            ax[0].set_title(f"Log count of {feature_name} shops in oa")
            ax[1].set_title(
                f"Log count of {feature_name} shops in {access.small_search_radius}km radius"
            )
            ax[2].set_title(
                f"Log count of {feature_name} shops in {access.large_search_radius}km radius"
            )
        else:
            ax[0].set_title(f"Log count of {feature_name}s in oa")
            ax[1].set_title(
                f"Log count of {feature_name}s in {access.small_search_radius}km radius"
            )
            ax[2].set_title(
                f"Log count of {feature_name}s in {access.large_search_radius}km radius"
            )
    plt.tight_layout()


task_1_summary_sql_query = """
SELECT
oa_geo_data.oa21cd as oa21cd,
oa_geo_data.latitude,
oa_geo_data.longitude,
ts062_data.Total_all_usual_residents,
ts062_data.l15 AS number_of_students,
ts011_data.1_deprivation,
ts011_data.2_deprivation,
ts011_data.3_deprivation,
ts011_data.4_deprivation,
ts011_data.total_households,
polygon_counts.historic AS exact_historic, polygon_counts.bar AS exact_bar, polygon_counts.cafe AS exact_cafe, polygon_counts.restaurant AS exact_restaurant, polygon_counts.college AS exact_college, polygon_counts.library AS exact_library, polygon_counts.school AS exact_school,
polygon_counts.university AS exact_university, polygon_counts.fuel AS exact_fuel, polygon_counts.bank AS exact_bank, polygon_counts.doctors AS exact_doctors, polygon_counts.hospital AS exact_hospital,
polygon_counts.church AS exact_church, polygon_counts.mosque AS exact_mosque, polygon_counts.synagogue AS exact_synagogue, polygon_counts.leisure AS exact_leisure, polygon_counts.supermarket AS exact_supermarket,
polygon_counts.convenience AS exact_convenience, polygon_counts.department_store AS exact_department_store, polygon_counts.clothes AS exact_clothes, polygon_counts.charity AS exact_charity, polygon_counts.books AS exact_books, polygon_counts.sport AS exact_sport,
small_radius_counts.historic AS small_historic, small_radius_counts.bar AS small_bar, small_radius_counts.cafe AS small_cafe, small_radius_counts.restaurant AS small_restaurant, small_radius_counts.college AS small_college, small_radius_counts.library AS small_library, small_radius_counts.school AS small_school,
small_radius_counts.university AS small_university, small_radius_counts.fuel AS small_fuel, small_radius_counts.bank AS small_bank, small_radius_counts.doctors AS small_doctors, small_radius_counts.hospital AS small_hospital,
small_radius_counts.church AS small_church, small_radius_counts.mosque AS small_mosque, small_radius_counts.synagogue AS small_synagogue, small_radius_counts.leisure AS small_leisure, small_radius_counts.supermarket AS small_supermarket,
small_radius_counts.convenience AS small_convenience, small_radius_counts.department_store AS small_department_store, small_radius_counts.clothes AS small_clothes, small_radius_counts.charity AS small_charity,
small_radius_counts.books AS small_books, small_radius_counts.sport AS small_sport,
large_radius_counts.historic AS large_historic, large_radius_counts.bar AS large_bar, large_radius_counts.cafe AS large_cafe, large_radius_counts.restaurant AS large_restaurant, large_radius_counts.college AS large_college, large_radius_counts.library AS large_library, large_radius_counts.school AS large_school,
large_radius_counts.university AS large_university, large_radius_counts.fuel AS large_fuel, large_radius_counts.bank AS large_bank, large_radius_counts.doctors AS large_doctors, large_radius_counts.hospital AS large_hospital,
large_radius_counts.church AS large_church, large_radius_counts.mosque AS large_mosque, large_radius_counts.synagogue AS large_synagogue, large_radius_counts.leisure AS large_leisure,
large_radius_counts.supermarket AS large_supermarket,
large_radius_counts.convenience AS large_convenience, large_radius_counts.department_store AS large_department_store, large_radius_counts.clothes AS large_clothes, large_radius_counts.charity AS large_charity,
large_radius_counts.books AS large_books, large_radius_counts.sport AS large_sport
FROM `oa_geo_data`
JOIN `polygon_counts` ON `polygon_counts`.`oa21cd` = `oa_geo_data`.`oa21cd`
JOIN `small_radius_counts` ON `polygon_counts`.`oa21cd` = `small_radius_counts`.`oa21cd`
JOIN `large_radius_counts` ON `polygon_counts`.`oa21cd` = `large_radius_counts`.`oa21cd`
JOIN `ts062_data` ON `ts062_data`.`geography_code` = `oa_geo_data`.`oa21cd`
JOIN `ts011_data` ON `ts062_data`.`geography_code` = `ts011_data`.`geography_code`
ORDER BY RAND() LIMIT 2500;
"""


def generate_task2_summary_query():
    sql_query = "SELECT\n"
    for column in access.column_list:
        for size in size_table_name_map:
            sql_query += (
                f" {size_table_name_map[size]}_counts.{column} as {size}_{column},\n"
            )
    for size in size_table_name_map:
        for column in access.column_list:
            sql_query += f" {size_table_name_map[size]}_counts.{column} +"
        sql_query = sql_query[:-1]
        sql_query += f"as {size}_all_categories,\n"
    for year, table in occupations_tables_map.items():
        for occupation in access.occupations_list[1:]:
            sql_query += f" {table}.{occupation} as {occupation}_{year},\n"
    sql_query += " oa_geo_data.oa21cd as oa21cd\n"
    sql_query += " FROM oa_geo_data\n"
    for size in size_table_name_map.values():
        sql_query += (
            f" JOIN {size}_counts ON oa_geo_data.oa21cd = {size}_counts.oa21cd\n"
        )
    for table in occupations_tables_map.values():
        sql_query += f" JOIN {table} ON oa_geo_data.oa21cd = {table}.oa_2021\n"
    sql_query += "ORDER BY RAND()\n LIMIT 5000;"
    return sql_query


def process_t1_sample(sample):
    column_list = access.column_list
    sample_df = pd.DataFrame(sample)
    for size in access.size_list:
        sample_df.drop(f"{size}_church", axis=1)
    sample_df["Average_deprivation"] = (
        sample_df["1_deprivation"]
        + 2 * sample_df["2_deprivation"]
        + 3 * sample_df["3_deprivation"]
        + 4 * sample_df["4_deprivation"]
    ) / sample_df["total_households"]
    sample_df["Percentage_of_students"] = (
        sample_df["number_of_students"] / sample_df["Total_all_usual_residents"]
    )
    sample_df.set_index("oa21cd", inplace=True)
    norm_df = sample_df.copy()
    log_df = sample_df.copy()
    for size in access.size_list:
        sized_column_names = list(map(lambda column: f"{size}_{column}", column_list))
        sums = sample_df[sized_column_names].sum(axis=1)
        norm_df[sized_column_names] = sample_df[sized_column_names].div(sums, axis=0)
        log_df[sized_column_names] = sample_df[sized_column_names].apply(
            lambda column: np.log10(column + 1)
        )
    return {"true": sample_df, "normalised": norm_df, "log": log_df}, {
        "Percentage of students": sample_df["Percentage_of_students"],
        "Average Deprivation": sample_df["Average_deprivation"],
    }


def process_t2_sample(sample):
    df = pd.DataFrame(sample)
    normalised_df = df.copy()
    normalised_df.set_index("oa21cd", inplace=True)
    for size in size_table_name_map:
        for column in access.column_list:
            normalised_df[f"{size}_{column}"] = (
                normalised_df[f"{size}_{column}"]
                / normalised_df[f"{size}_all_categories"]
            )
            normalised_df.loc[
                normalised_df[f"{size}_{column}"].isna(), f"{size}_{column}"
            ] = 0
    for occupation in access.occupations_list[1:]:
        normalised_df[f"{occupation}_change"] = (
            normalised_df[f"{occupation}_{2021}"]
            - normalised_df[f"{occupation}_{2001}"]
        ) / normalised_df[f"{occupation}_{2001}"].astype(float)
        normalised_df.loc[
            normalised_df[f"{occupation}_change"].isna(), f"{occupation}_change"
        ] = 1
        normalised_df.loc[
            normalised_df[f"{occupation}_change"] == np.inf, f"{occupation}_change"
        ] = 1
    sized_dfs = {
        size: normalised_df[[f"{size}_{column}" for column in access.column_list]]
        for size in size_table_name_map
    }
    response_vectors = {
        occupation: normalised_df[f"{occupation}_change"]
        for occupation in access.occupations_list[1:]
    }
    return sized_dfs, response_vectors


def display_correlation_heatmaps(dfs):
    ncols = len(dfs)
    nrows = len(access.size_list)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13 * ncols, 13 * nrows))
    for (df_name, df), ax_row in zip(dfs.items(), axs):
        for size, ax in zip(access.size_list, ax_row):
            sized_column_names = list(
                map(lambda column: f"{size}_{column}", access.column_list)
            )
            sized_df = df[sized_column_names]
            display_heatmap(
                sized_df.corr(),
                title=f"Correlation of {df_name} feature counts in {size} area around oa",
                ax=ax,
            )
    plt.show()


def display_feature_correlations(dfs, response_vectors):
    ncols = len(dfs)
    nrows = len(response_vectors)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12 * ncols, 20 * nrows))
    for (response_vector_name, response_vector), ax_row in zip(
        response_vectors.items(), axs
    ):
        for (df_name, df), ax in zip(dfs.items(), ax_row):
            corrs = {
                size: {
                    column: df[f"{size}_{column}"].corr(response_vector)
                    for column in access.column_list
                }
                for size in access.size_list
            }
            corrs_df = pd.DataFrame(corrs)
            display_heatmap(
                corrs_df,
                title=f"Correlation of {df_name} feature_counts against {response_vector_name} in area around oa",
                ax=ax,
                use_rows=True,
            )


def display_feature_correlations_on_one_graph(
    df, response_vectors, df_name="features", response_vectors_name="response vectors"
):
    corrs = {
        response_vector_name: {
            feature: feature_series.corr(response_vector)
            for feature, feature_series in df.items()
        }
        for response_vector_name, response_vector in response_vectors.items()
    }
    corrs_df = pd.DataFrame(corrs)
    display_heatmap(
        df=corrs_df,
        title=f"Correlation between {response_vectors_name} and {df_name}",
        use_rows=True,
    )


def display_single_response_vector_histogram(
    response_vector_name, response_vector, ax, min_value=None, max_value=None
):
    if min_value is None:
        min_value = min(response_vector.min(), 0)
    if max_value is None:
        max_value = response_vector.max()
    ax.hist(
        response_vector,
        bins=np.arange(min_value, max_value, (max_value - min_value) / 100),
    )
    ax.set_title(f"Distribution of {response_vector_name} across output areas")


def display_response_vector_histogram(response_vectors, min_value=None, max_value=None):
    nrows = len(response_vectors)
    fig, axs = plt.subplots(nrows=nrows, figsize=(10, nrows * 10))
    for (response_vector_name, response_vector), ax in zip(
        response_vectors.items(), axs
    ):
        display_single_response_vector_histogram(
            response_vector_name, response_vector, ax, min_value, max_value
        )


def display_t1_features_vector_summaries(conn):
    deprivation_data = access.sql_select(
        conn,
        "SELECT (1_deprivation + 2 * 2_deprivation + 3 * 3_deprivation + 4 * 4_deprivation) / total_households AS average_deprivation FROM `ts011_data`",
    )
    deprivation_df = pd.DataFrame(deprivation_data, columns=["Average Deprivation"])
    students_data = access.sql_select(
        conn,
        "SELECT l15 / total_all_usual_residents AS 'Percentage_of_students' FROM `ts062_data`",
    )
    students_df = pd.DataFrame(students_data, columns=["Percentage of Students"])
    response_vectors = {
        "Percentage of Students": students_df["Percentage of Students"],
        "Average Deprivation": deprivation_df["Average Deprivation"],
    }
    display_response_vector_histogram(response_vectors)


def display_t2_features_vector_summaries(conn):
    response_vectors = {}
    for occupation in access.occupations_list[1:]:
        occupation_sql = f"""
        SELECT
        (occupations_2021_data.{occupation} - occupations_2001_data_rezoned.{occupation}) / occupations_2001_data_rezoned.{occupation}
        FROM occupations_2001_data_rezoned JOIN occupations_2021_data ON occupations_2021_data.oa_2021 = occupations_2001_data_rezoned.oa_2021;"""
        occupation_data = access.sql_select(conn, occupation_sql)
        occupation_df = pd.DataFrame(occupation_data, columns=["Change"])
        response_vectors[occupation] = occupation_df["Change"]
    display_response_vector_histogram(response_vectors, min_value=-1, max_value=2)


def fit_exploratory_models(dfs, response_vectors):
    for response_vector_name, response_vector in response_vectors.items():
        for size in access.size_list:
            sized_column_names = list(
                map(lambda column: f"{size}_{column}", access.column_list)
            )
            for name, df in dfs.items():
                design_matrix = df.dropna(subset=sized_column_names)
                model = sm.OLS(
                    response_vector[design_matrix.index],
                    sm.add_constant(design_matrix[sized_column_names]),
                )
                results = model.fit()
                title = f"Model for {response_vector_name} against size {size} on {name} osm data"
                under_line = "=" * len(title)
                print(title)
                print(under_line)
                print(results.summary(), "\n\n\n\n\n")
        sized_column_names = []
        for column in access.column_list:
            sized_column_names += list(
                map(lambda size: f"{size}_{column}", access.size_list)
            )
        for name, df in dfs.items():
            design_matrix = df.dropna(subset=sized_column_names)
            model = sm.OLS(
                response_vector[design_matrix.index],
                sm.add_constant(design_matrix[sized_column_names]),
            )
            results = model.fit()
            title = (
                f"Model for {response_vector_name} against all sizes on {name} osm_data"
            )
            under_line = "=" * len(title)
            print(title)
            print(under_line)
            print(results.summary(), "\n\n\n\n\n")


def get_nationwide_occupation_sql(table):
    return f"SELECT SUM(`all_categories`), SUM(`managers`), SUM(`professional_occupations`), SUM(`associate_professional_occupations`), SUM(`administrative_occupations`), SUM(`skilled_trades`), SUM(`personal_service`), SUM(`sales_and_customer_service`), SUM(`process_manufacturing`), SUM(`elementary_occupations`) FROM `{table}`;"


def display_nationwide_occupation_data(total_2001_data, total_2021_data):
    total_2021_df = pd.DataFrame(total_2021_data)
    total_2001_df = pd.DataFrame(total_2001_data)
    fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
    axs[0].bar(access.occupations_list[1:], total_2021_df.iloc[0][1:])
    axs[0].set_title("2021")
    axs[0].set_xlabel("Occupation")
    axs[0].set_ylabel("Count")
    axs[0].set_xticklabels(access.occupations_list[1:], rotation=90)
    axs[1].bar(access.occupations_list[1:], total_2001_df.iloc[0][1:])
    axs[1].set_title("2001")
    axs[1].set_xlabel("Occupation")
    axs[1].set_ylabel("Count")
    axs[1].set_xticklabels(access.occupations_list[1:], rotation=90)
    changes = (
        total_2021_df.iloc[0][1:] - total_2001_df.iloc[0][1:]
    ) / total_2001_df.iloc[0][1:]
    axs[2].bar(access.occupations_list[1:], changes)
    axs[2].set_title("Change in occupation count")
    axs[2].set_xlabel("Occupation")
    axs[2].set_ylabel("Change")
    axs[2].set_xticklabels(access.occupations_list[1:], rotation=90)
    plt.show()


def display_kmeans_elbows(dfs):
    ncols = len(dfs)
    fig, axes = plt.subplots(ncols=ncols, figsize=(10 * ncols, 10))
    for (df_name, df), ax in zip(dfs.items(), axes):
        inertias = []
        for k in range(3, len(df.columns)):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(df.T)
            inertias.append(kmeans.inertia_)
        ax.plot(range(3, len(df.columns)), inertias, marker="o")
        ax.set_title(f"{df_name} Radius Intertias")
        ax.set_xlabel("Number of clusters")
        ax.set_xticks(range(3, len(df.columns)))
        ax.set_ylabel("Inertia")
    plt.show()


def generate_clusters(dfs, ks_dict):
    features = []
    arbitrary_index_column = random.choice(list(dfs.values())).iloc[:,0]
    clusters_df = pd.DataFrame(arbitrary_index_column, columns="index_column")
    for df_name, df in dfs.items():
        kmeans = KMeans(n_clusters=ks_dict[df_name], random_state=0).fit(df.T)
        clusters = [[] for _ in range(ks_dict[df_name])]
        for i, cluster in enumerate(kmeans.labels_):
            clusters[cluster].append(df.columns[i])
        print("Clusters for", df_name, "radius:")
        for cluster in clusters:
            cluster_name = f"{cluster[0]}_cluster"
            print(f"{cluster_name}: {cluster}")
            clusters_df[cluster_name] = 0
            for column in cluster:
                clusters_df[cluster_name] += df[column]
        features += clusters
        print()
    clusters_df.drop(labels="index_column", inplace=True)
    return features, clusters_df
