#!/usr/bin/env geo_env2

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares (13-01-2026)
#     Created by Etienne Kras (etienne.kras@deltares.nl)
#   --------------------------------------------------------------------

# %% load packages and set paths

# packages
import os
import pandas as pd
import numpy as np

# paths
results = r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results"
scriptname = "XX_MVSlope_PostProcess"

if not os.path.exists(os.path.join(results, scriptname)):
    os.mkdir(os.path.join(results, scriptname))

# %% Load global transects to classify
global_transects = pd.read_csv(
    r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Data\GlobalMVTransectsForClassification_incSM.csv",
    delimiter=",",
)

# global_transects = global_transects.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
global_transects = global_transects.drop(columns=["index"])

# Add box_ids
boxes = []
for b in range(len(global_transects)):
    box_id = global_transects["transect_id"][b][0:11]
    boxes.append(box_id)
global_transects["box_id"] = boxes

box_unique = pd.DataFrame(np.unique(boxes))
box_shuffle = box_unique.sample(frac=1, random_state=1).reset_index(
    drop=True
)  # shuffled order

# check progress of processed boxes
all_files = []
for _, _, files in os.walk(
    os.path.join(results, "XX_HybridTransectModel_perTransectFastFast_MVSlope")
):
    if ".csv" in str(files):
        all_files.extend(files)
print("Number of processed boxes: %s of %s" % (len(all_files), len(box_unique)))

# make one large df and check progress w.r.t. processed transects
all_transects = pd.DataFrame()
for box in range(len(box_shuffle)):
    if box_shuffle[0][box] + "_ClassifiedTransects2020.csv" in all_files:
        transect_df = pd.read_csv(
            os.path.join(
                results,
                "XX_HybridTransectModel_perTransectFastFast_MVSlope",
                box_shuffle[0][box][0:7],
                "CSVperBOX",
                box_shuffle[0][box] + "_ClassifiedTransects2020.csv",
            ),
            delimiter=",",
        )
        last_proc_box_name = box_shuffle[0][box]
        # print("Length transect df (%s): %s" % (last_proc_box_name, len(transect_df)))
        all_transects = pd.concat([all_transects, transect_df], ignore_index=True)

print("Length last processed box (%s): %s" % (last_proc_box_name, len(transect_df)))
print(
    "Number of processed transects: %s of %s"
    % (len(all_transects), len(global_transects))
)

all_transects = all_transects.drop(columns=["Unnamed: 0"])

# store as CSV
all_transects.to_csv(
    os.path.join(
        results,
        scriptname,
        "Partly_Processed_Transects_MVSlope_V2.csv",
    ),
    index=False,
)

# %% load CSV swiftly to test config
# load global transects with SM data
store_fol = os.path.join(
    r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results", scriptname
)
df_name = "Partly_Processed_Transects_MVSlope_V2.csv"

df = pd.read_csv(os.path.join(store_fol, df_name))
