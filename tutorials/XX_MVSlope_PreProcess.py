#!/usr/bin/env geo_env2

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares (11-11-2025)
#     Etienne Kras
#     etienne.kras@deltares.nl
#   --------------------------------------------------------------------

# %% load packages and set paths

# packages
import os
import pandas as pd
import numpy as np
import ast

# inspiration scripts
# - p:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Scripts\TransectProfile.py
# - p:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Scripts\HybridTransectModel_perTransectFastFast.py

# Script workflow
# 	V Load global transects (filter on muddy & vegetated shorelines from Romys 2020)
# 	V Load Shoreline Monitor SDS point for 2020 (along transects) - and if no data or differently, use other closest point in time.

# paths
mc_data_fol = r"p:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Data"  # muddy coast data folder
SM_data_fol = r"p:\1000545-054-globalbeaches\04_Shoreline_Monitor_data_requests\Data_requests\ShorelineMonitor"
store_fol = r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Data"

# datasets
ct_data = r"WorldClassifiedTransects2020.csv"  # classified transects dataset (1.826.995 transects)
SM_data = r"ShorelineMonitor_1984_2016_v1.1_set1.csv"  # original SM dataset
SM_red_data = (
    r"ShorelineMonitor_1984_2016_v1.1_set1_adjusted_dt_dist.csv"  # adjusted SM dataset
)
hc_data = r"TransectTrainingDataframe.csv"  # hybrid classier training dataset
fin_data = r"GlobalMVTransectsForClassification_incSM.csv"  # final prepared dataset as input to classification and slope calc

# settings
t0 = 1984
preprocess_SM = False  # put to true if preprocessing SM dataset, if file already available set to False

# %% load global transects & filter on muddy and vegetated shorelines from Romy Hulskamp (https://www.nature.com/articles/s41467-023-43819-6)

# open CSV file (where 0=sandy, 1=muddy, 2=rocky, 3=vegetated, 4=other)
ct_df = pd.read_csv(os.path.join(mc_data_fol, ct_data))

# %% load Shoreline Monitor SDS point and do modifications
if preprocess_SM == True:  # pre-process data
    SM_df = pd.read_csv(os.path.join(SM_data_fol, SM_data), delimiter=",")

    # reduce the dataset keeping only transect_id, changerate, dist, dt and outliers
    SM_df = SM_df[
        ["transect_id", "changerate", "dt", "dist", "outliers_1", "outliers_2"]
    ]

    # convert columns from string to numpy array
    outliers_1 = SM_df["outliers_1"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=int)
    )
    outliers_2 = SM_df["outliers_2"].apply(
        lambda x: np.array(ast.literal_eval(x), dtype=int)
    )
    dt = SM_df["dt"].apply(lambda x: np.array(ast.literal_eval(x)))
    dist = SM_df["dist"].apply(lambda x: np.array(ast.literal_eval(x)))

    # start to remove
    dt_clean_red = [np.delete(a, idx) for a, idx in zip(dt, outliers_1)]
    dist_clean_red = [np.delete(a, idx) for a, idx in zip(dist, outliers_1)]
    dt_clean_red = [np.delete(a, idx) for a, idx in zip(dt_clean_red, outliers_2)]
    dist_clean_red = [np.delete(a, idx) for a, idx in zip(dist_clean_red, outliers_2)]

    # round all in dt_clean_red and add t0
    dt_red = [np.round(a).astype(int) + t0 for a in dt_clean_red]  # round all values

    # select the SDS point for 2020 (dt=2020), or the closest available year
    # Note: this is always 2016, or the last dt point, for this SM dataset that goes untill 2016
    dt_red_last = [a[-1] if len(a) > 0 else None for a in dt_red]
    dist_red_last = [a[-1] if len(a) > 0 else None for a in dist_clean_red]

    # re-adding to SM_df with last numbers in timeseries
    SM_df_red = SM_df[["transect_id", "changerate"]]
    SM_df_red["dt_last"] = dt_red_last
    SM_df_red["dist_last"] = dist_red_last

    # save reduced SM dataset
    SM_df_red.to_csv(os.path.join(store_fol, SM_red_data), index=False)
    SM_df = SM_df_red  # replace

if preprocess_SM == False:  # load directly
    # load SM reduced dataset
    SM_df = pd.read_csv(os.path.join(store_fol, SM_red_data))

# %% finalising pre-work

# match SM dataset and muddy dataset on transect ID
ct_SM_df = pd.merge(ct_df, SM_df, on="transect_id", how="inner")

# filter muddy & vegetated transects
ct_df_mv = ct_SM_df[((ct_SM_df["prediction"] == 1) | (ct_SM_df["prediction"] == 3))]
ct_df_mv = ct_df_mv.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2"])

# print some stats
print(
    "percentage sandy transects",
    round(ct_df[(ct_df["prediction"] == 0)].shape[0] / len(ct_df) * 100, 1),
)  # 25.7
print(
    "percentage muddy transects",
    round(ct_df[(ct_df["prediction"] == 1)].shape[0] / len(ct_df) * 100, 1),
)  # 12.1
print(
    "percentage rocky transects",
    round(ct_df[(ct_df["prediction"] == 2)].shape[0] / len(ct_df) * 100, 1),
)  # 26.8
print(
    "percentage vegetated transects",
    round(ct_df[(ct_df["prediction"] == 3)].shape[0] / len(ct_df) * 100, 1),
)  # 33.7
print(
    "percentage other transects",
    round(ct_df[(ct_df["prediction"] == 4)].shape[0] / len(ct_df) * 100, 1),
)  # 1.7
print(
    "percentage muddy & vegetated transects", round(len(ct_df_mv) / len(ct_df) * 100, 1)
)  # 45.7

# save dataset
if preprocess_SM == True:  # load directly
    ct_df_mv.to_csv(os.path.join(store_fol, fin_data), index=False)
