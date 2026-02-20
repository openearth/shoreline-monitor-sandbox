#!/usr/bin/env geo_env2

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares (17-02-2026)
#     Created by Etienne Kras (etienne.kras@deltares.nl)
#   --------------------------------------------------------------------

# %% load packages and set paths

# packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# paths
results = r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results"
scriptname = "XX_MVSlope_GlobStats"

if not os.path.exists(os.path.join(results, scriptname)):
    os.mkdir(os.path.join(results, scriptname))

# %% Load global transects to classify
# global_transects = pd.read_csv(
#     r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results\XX_MVSlope_PostProcess\Partly_Processed_Transects_MVSlope_V3.csv",
#     delimiter=",",
# )

# global_transects = global_transects.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
# global_transects = global_transects.drop(columns=["index"])

store_fol = (
    r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results\XX_MVSlope_PostProcess"
)
df_name = "Complete_Processed_Transects_MVSlope.csv"

gtr = pd.read_csv(os.path.join(store_fol, df_name))

# %% derive statistics; per continent & per country, per type..

# statistics on the slope in a continent (mean, median, std, min, max, 5th and 95th percentile)
gtr_continent_ms_stats = (
    gtr.groupby("continent")["mud_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)
gtr_continent_vs_stats = (
    gtr.groupby("continent")["veg_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)

# statistics on the slope in a country (mean, median, std, min, max, 5th and 95th percentile)
gtr_country_ms_stats = (
    gtr.groupby("country_name")["mud_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)
gtr_country_vs_stats = (
    gtr.groupby("country_name")["veg_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)

# %% per type (prediction) & distribution of slope values (mean, median, std, min, max, 5th and 95th percentile)

gtr_pred_ms_stats = (
    gtr.groupby("prediction")["mud_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)

# plot prediction distribution for muddy slope for prediction == 1 (mud)
plt.figure(figsize=(10, 6))
plt.hist(
    gtr[gtr["prediction"] == 1]["mud_slope"],
    bins=100,
    edgecolor="black",
    range=(1e-10, 0.1),
)
plt.title("Distribution of Muddy Slope for Prediction == 1 (Mud)")
plt.show()

gtr_pred_vs_stats = (
    gtr.groupby("prediction")["veg_slope"]
    .agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .sort_values(by="mean", ascending=True)
)

# plot prediction distribution for vegetated slope for prediction == 3 (vegetated)
plt.figure(figsize=(10, 6))
plt.hist(
    gtr[gtr["prediction"] == 3]["veg_slope"],
    bins=100,
    edgecolor="black",
    range=(1e-10, 0.3),
)
plt.title("Distribution of Vegetated Slope for Prediction == 3 (Vegetated)")
plt.show()


# TODO: ideas to match slope on IPCC region
# TODO: test if we can complete the database


# %% Load global transects to classify and check progress w.r.t. processed transects
global_transects = pd.read_csv(
    r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Data\GlobalMVTransectsForClassification_incSM.csv",
    delimiter=",",
)

# global_transects = global_transects.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
global_transects = global_transects.drop(columns=["index"])

# count occurrences of prediction values
prediction_counts = global_transects["prediction"].value_counts()
