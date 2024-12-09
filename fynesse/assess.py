from . import access
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np


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
oa_geo_data.oa21cd,
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


def process_t1_sample(sample):
    column_list = access.column_list
    sample_df = pd.DataFrame(sample)
    for size in access.size_list:
        sample_df.drop(f"{size}_church", axis=1)
    column_list.remove("church")
    sample_df["Average_deprivation"] = (
        sample_df["1_deprivation"]
        + 2 * sample_df["2_deprivation"]
        + 3 * sample_df["3_deprivation"]
        + 4 * sample_df["4_deprivation"]
    ) / sample_df["total_households"]
    sample_df["Percentage_of_students"] = (
        sample_df["number_of_students"] / sample_df["Total_all_usual_residents"]
    )
    norm_df = sample_df.copy()
    log_df = sample_df.copy()
    for size in access.size_list:
        sized_column_names = list(map(lambda column: f"{size}_{column}", column_list))
        sums = sample_df[sized_column_names].sum(axis=1)
        norm_df[sized_column_names] = sample_df[sized_column_names].div(sums, axis=0)
        log_df[sized_column_names] = sample_df[sized_column_names].apply(
            lambda column: np.log10(column + 1)
        )
    return {"true": sample_df, "normalised": norm_df, "log": log_df}


def display_correlation_heatmaps(dfs):
    for df_name, df in dfs.items():
        fig, axes = plt.subplots(nrows=len(access.size_list), figsize=(10, 40))
        for size, ax in zip(access.size_list, axes):
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
