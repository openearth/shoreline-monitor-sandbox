#!/usr/bin/env geo_env2

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares (30-03-2022)
#     Created by Romy Hulskamp
#     Modified by Etienne Kras (etienne.kras@deltares.nl)
#   --------------------------------------------------------------------

import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import datetime
import os.path
import json
import time
import pandas as pd
import os
import math
import csv
import statistics
import sklearn
from sklearn.ensemble import RandomForestClassifier
from geojson import MultiPoint, Feature, FeatureCollection, dump, LineString
import warnings
import cartopy.io.img_tiles as cimgt
import contextily as ctx

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None
import ee

# GEE specific packages
project = "shorelines-11208011-022"

try:
    ee.Initialize(project=project)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project)

# get filename
folder = (
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Scripts"
)
results = r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Results"
figures = r"p:\1000545-054-globalbeaches\19_Muddy_Slopes\Figures"
scriptname = "XX_HybridTransectModel_perTransectFastFast_MVSlope"
# outputfolder
os.chdir(folder)
if not os.path.exists(os.path.join(results, scriptname)):
    os.mkdir(os.path.join(results, scriptname))
    os.mkdir(os.path.join(figures, scriptname))

# %% define collection and bands
collection = "IM_S2"
collection_name = {"IM_S2": "COPERNICUS/S2_HARMONIZED"}
band_names = {
    "IM_S2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "QA60",
    ]
}
band_std_names = [
    "Aerosols",
    "Blue",
    "Green",
    "Red",
    "Red Edge 1",
    "Red Edge 2",
    "Red Edge 3",
    "NIR",
    "Red Edge 4",
    "SWIR 1",
    "SWIR 2",
    "QA60",
]
band_std_names_mean = [
    "Aerosols_mean",
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "Red Edge 1_mean",
    "Red Edge 2_mean",
    "Red Edge 3_mean",
    "NIR_mean",
    "Red Edge 4_mean",
    "SWIR 1_mean",
    "SWIR 2_mean",
    "QA60_mean",
]
band_names_classifier = [
    "Aerosols",
    "Blue",
    "Green",
    "Red",
    "Red Edge 1",
    "Red Edge 2",
    "Red Edge 3",
    "NIR",
    "Red Edge 4",
    "SWIR 1",
    "SWIR 2",
    "NDWI",
    "NDVI",
]
# select training dates
date_range_training = ["2020-01-01", "2020-12-31"]
sdate = date_range_training[0]
edate = date_range_training[1]
sdate = datetime.datetime.strptime(sdate, "%Y-%m-%d")
edate = datetime.datetime.strptime(edate, "%Y-%m-%d")
tempsdate = ee.Date(sdate)
tempedate = ee.Date(edate)
# set cloudcover limit
cloudcover_limit = 15
# # distance between extracted points in meter
dist_steps = 10
scale = 10
length = 1500

# %% Load multispectral image classification model
pixel_training = pd.read_csv(
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Data\PixelTrainingDataframe.csv",
    delimiter=",",
)
featuresjson = []
for n in range(len(pixel_training)):
    point = MultiPoint([(1, 1)])
    featuresjson.append(
        Feature(
            geometry=point,
            properties={
                "Point_id": str(n),
                "Aerosols": (pixel_training["Aerosols"][n]),
                "Blue": (pixel_training["Blue"][n]),
                "Green": (pixel_training["Green"][n]),
                "Red": (pixel_training["Red"][n]),
                "Red Edge 1": (pixel_training["Red Edge 1"][n]),
                "Red Edge 2": (pixel_training["Red Edge 2"][n]),
                "Red Edge 3": (pixel_training["Red Edge 3"][n]),
                "NIR": (pixel_training["NIR"][n]),
                "Red Edge 4": (pixel_training["Red Edge 4"][n]),
                "SWIR 1": (pixel_training["SWIR 1"][n]),
                "SWIR 2": (pixel_training["SWIR 2"][n]),
                "NDWI": (pixel_training["NDWI"][n]),
                "NDVI": (pixel_training["NDVI"][n]),
                "class": (pixel_training["num"][n]),
            },
        )
    )
feature_collection = FeatureCollection(featuresjson)

# Train image classifier
classifier_image = ee.Classifier.smileRandomForest(15).train(
    feature_collection, "class", band_names_classifier
)
classifier_image = classifier_image.setOutputMode(mode="CLASSIFICATION")

# %% Load hybrid classification model
transect_training = pd.read_csv(
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Data\TransectTrainingDataframe.csv"
)

# Train transect classifier
features = [
    "mud",
    "height max",
    "height var",
    "water",
    "sand",
    "gsw",
    "abs lat",
    "turbid",
    "mintemp",
    "tidal range",
    "maxtemp",
    "Center_lat",
    "vegetation",
    "other",
    "dry",
    "intertidal",
    "mangrove",
]
tt_f = transect_training.loc[:, features]
tt_norm = (tt_f - tt_f.min()) / (tt_f.max() - tt_f.min())
tt_label = transect_training["label"].map(
    {"sandy coast": 0, "mud coast": 1, "rocky coast": 2, "vegetated": 3, "other": 4}
)
rfc_hybrid = RandomForestClassifier(n_estimators=15, random_state=0)
classifier_hybrid = rfc_hybrid.fit(tt_norm, tt_label)

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

global_transects["abs lat"] = abs(global_transects["Center_lat"])

box_unique = pd.DataFrame(np.unique(boxes))
box_shuffle = box_unique.sample(frac=1, random_state=1).reset_index(
    drop=True
)  # shuffled order

# choosebox = ["BOX_117_039"]  # suriname: transects set t in line 151 -> range(0,5)
# # choosebox = ["BOX_079_005"]  # madagascar: transects set t in line 151 -> range(20,25)
# box_shuffle = pd.DataFrame(choosebox)


# %% definitions
# bicubic sampling
def bicubic(image):
    return image.resample("bicubic").focal_median(1)


# interpolation
def get_nearest_value(f, image):
    buff = f.geometry().buffer(20000, 1000)
    distance = image.addBands(ee.FeatureCollection(f).distance())
    values = distance.sample(
        region=buff,
        scale=1000,
        numPixels=1000000,
        seed=42,
        dropNulls=True,
        geometries=True,
    )  # {'region': ee.Geometry(buff), 'scale':1000, 'numPixels': 1000000, 'seed':42, 'dropNulls': True})
    values = values.sort("distance", True)
    return ee.Algorithms.If(
        values.size().eq(0),
        ee.Feature(None).set(image.bandNames().get(0), -999),
        values.first(),
    )


# %% Hybrid Transect classification
for box in range(len(box_shuffle)):

    # Create folder structure
    if not os.path.exists(os.path.join(results, scriptname, box_shuffle[0][box][0:7])):
        os.mkdir(os.path.join(results, scriptname, box_shuffle[0][box][0:7]))
    if not os.path.exists(
        os.path.join(results, scriptname, box_shuffle[0][box][0:7], "JSONperBOX")
    ):
        os.mkdir(
            os.path.join(results, scriptname, box_shuffle[0][box][0:7], "JSONperBOX")
        )
    if not os.path.exists(
        os.path.join(results, scriptname, box_shuffle[0][box][0:7], "CSVperBOX")
    ):
        os.mkdir(
            os.path.join(results, scriptname, box_shuffle[0][box][0:7], "CSVperBOX")
        )

    # Check if box already has been classified
    if not os.path.exists(
        os.path.join(
            results,
            scriptname,
            box_shuffle[0][box][0:7],
            "JSONperBOX",
            box_shuffle[0][box] + "_ClassifiedTransects2020.json",
        )
    ):

        transect_df = global_transects[
            global_transects["box_id"] == box_shuffle[0][box]
        ].reset_index(drop=True)
        transect_df[
            [
                "abs lat",
                "sand",
                "mud",
                "water",
                "vegetation",
                "other",
                "turbid",
                "dry",
                "height max",
                "height var",
                "mangrove",
                "intertidal",
                "gsw",
                "maxtemp",
                "mintemp",
                "mhhw",
                "mllw",
                "tidal range",
                "prediction",
                "dem profile",
                "veg_slope",
                "mud_slope",
            ]
        ] = None
        transect_df["abs lat"] = abs(transect_df["Center_lat"])

        print(box, box_shuffle[0][box], len(transect_df), "transects")

        begin = time.time()

        ### per box; additional variables

        batch = 250  # set maximum batch size (max 250 transects at once)
        for r in range(0, len(transect_df), batch):
            if r + batch > len(transect_df):
                last = len(transect_df)
            else:
                last = r + batch

            transect_points_box = []
            for t in range(r, last):
                transect_id = transect_df["transect_id"][t]
                f = open(
                    os.path.join(
                        r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Mud_RHulskamp",
                        "muddy results",
                        "2021-09-23_CreateTransects_withPython",
                        "Points",
                        transect_id + "_Points.json",
                    ),
                    "r",
                )
                transect_points = json.load(f)
                f.close()
                transect_points_box.append(transect_points)
            transect_feat_box = ee.FeatureCollection(
                [item for sublist in transect_points_box for item in sublist]
            )
            centers = ee.FeatureCollection(
                [
                    ee.Geometry.Point(
                        [transect_df["Center_lon"][t], transect_df["Center_lat"][t]]
                    )
                    for t in range(r, last)
                ]
            )

            dem = ee.Image("MERIT/DEM/v1_0_3")
            dem_sample = dem.unmask(-999, False).sampleRegions(
                transect_feat_box, ["id"], scale
            )
            dem_sample_info_T = dem_sample.aggregate_array("dem").getInfo()

            mangrove = ee.ImageCollection("LANDSAT/MANGROVE_FORESTS").first()
            mangrove_sample = mangrove.unmask(-999, False).sampleRegions(
                transect_feat_box, ["id"], scale
            )
            mangrove_sample_info_T = mangrove_sample.aggregate_array("1").getInfo()

            intertidal = ee.ImageCollection(
                "UQ/murray/Intertidal/v1_1/global_intertidal"
            ).first()
            intertidal_sample = intertidal.unmask(-999, False).sampleRegions(
                transect_feat_box, ["id"], scale
            )
            intertidal_sample_info_T = intertidal_sample.aggregate_array(
                "classification"
            ).getInfo()

            GSW = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select(
                "occurrence"
            )  # TODO: replace with GSW1_4 later
            GSW_sample = GSW.unmask(-999, False).sampleRegions(
                transect_feat_box, ["id"], scale
            )
            GSW_sample_info_T = GSW_sample.aggregate_array("occurrence").getInfo()

            temp = ee.Image("WORLDCLIM/V1/BIO")
            pts = centers.map(lambda i: get_nearest_value(i, temp.select("bio05")))
            maxtemp_mean = np.array(pts.aggregate_array("bio05").getInfo()) * 0.1
            pts = centers.map(lambda i: get_nearest_value(i, temp.select("bio06")))
            mintemp_mean = np.array(pts.aggregate_array("bio06").getInfo()) * 0.1

            transect_df.loc[r : last - 1, "maxtemp"] = (
                maxtemp_mean  # transect_df["maxtemp"][r:last] = maxtemp_mean
            )
            transect_df.loc[r : last - 1, "mintemp"] = (
                mintemp_mean  # transect_df["mintemp"][r:last] = mintemp_mean
            )

            tidal = ee.Image("projects/dgds-gee/gtsm/tidal_indicators")
            pts = centers.map(
                lambda i: get_nearest_value(i, tidal.select("mean_higher_high_water"))
            )
            mhhw_mean = pts.aggregate_array(
                "mean_higher_high_water"
            ).getInfo()  # generates a list of all point values, at the moment is only 1 value, but could be expanded easily to get centers as a feature collection
            pts = centers.map(
                lambda i: get_nearest_value(i, tidal.select("mean_lower_low_water"))
            )
            mllw_mean = pts.aggregate_array("mean_lower_low_water").getInfo()

            transect_df.loc[r : last - 1, "mhhw"] = (
                mhhw_mean  # transect_df["mhhw"][r:last] = mhhw_mean
            )
            transect_df.loc[r : last - 1, "mllw"] = (
                mllw_mean  # transect_df["mllw"][r:last] = mllw_mean
            )
            transect_df.loc[r : last - 1, "tidal range"] = (
                transect_df.loc[r : last - 1, "mhhw"]
                - transect_df.loc[r : last - 1, "mllw"]
            )

            for t in range(len(transect_points_box)):
                dem_sample_info = pd.DataFrame(
                    {"dem": dem_sample_info_T[t * 151 : t * 151 + 151]}
                ).replace({-999: np.nan})
                transect_df.loc[r + t, "height max"] = np.max(
                    dem_sample_info["dem"]
                )  # transect_df["height max"][r + t] = np.max(dem_sample_info["dem"])
                transect_df.loc[r + t, "height var"] = np.var(
                    dem_sample_info["dem"]
                )  # transect_df["height var"][r + t] = np.var(dem_sample_info["dem"])
                transect_df.at[r + t, "dem profile"] = dem_sample_info[
                    "dem"
                ].tolist()  # transect_df["dem profile"][r + t] = dem_sample_info["dem"].tolist()

                mangrove_sample_info = pd.DataFrame(
                    {"mangrove": mangrove_sample_info_T[t * 151 : t * 151 + 151]}
                )
                mangrove_count = mangrove_sample_info[
                    "mangrove"
                ].value_counts()  # if one point contains mangrove (1), then mangroves occur
                if mangrove_sample_info["mangrove"].isin([1]).any().any():
                    transect_df.loc[r + t, "mangrove"] = (
                        1  # transect_df["mangrove"][r + t] = 1
                    )
                else:
                    transect_df.loc[r + t, "mangrove"] = (
                        0  # transect_df["mangrove"][r + t] = 0
                    )

                intertidal_sample_info = pd.DataFrame(
                    {"intertidal": intertidal_sample_info_T[t * 151 : t * 151 + 151]}
                )
                intertidal_count = intertidal_sample_info[
                    "intertidal"
                ].value_counts()  # count points that contain intertidal flat (1)
                if intertidal_sample_info["intertidal"].isin([1]).any().any():
                    transect_df.loc[r + t, "intertidal"] = (
                        intertidal_count[1] * dist_steps
                    )  # transect_df["intertidal"][r + t] = intertidal_count[1] * dist_steps
                else:
                    transect_df.loc[r + t, "intertidal"] = (
                        0  # transect_df["intertidal"][r + t] = 0
                    )

                GSW_sample_info = pd.DataFrame(
                    {"gsw": GSW_sample_info_T[t * 151 : t * 151 + 151]}
                )
                gsw90 = []
                for g in range(len(GSW_sample_info["gsw"])):
                    if GSW_sample_info["gsw"][g] > 5 and GSW_sample_info["gsw"][g] < 95:
                        gsw90value = GSW_sample_info["gsw"][g]
                        gsw90.append(gsw90value)
                transect_df.loc[r + t, "gsw"] = (
                    len(gsw90) * dist_steps
                )  # transect_df["gsw"][r + t] = len(gsw90) * dist_steps

        ### per batch, image classification

        try:

            batch = 1  # set maximum batch size (max 100 transects at once)
            for r in range(0, len(transect_df), batch):
                if r + batch > len(transect_df):
                    last = len(transect_df)
                else:
                    last = r + batch

                transect_points_batch = []
                for t in range(r, last):
                    transect_id = transect_df["transect_id"][t]
                    f = open(
                        os.path.join(
                            r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Mud_RHulskamp",
                            "muddy results",
                            "2021-09-23_CreateTransects_withPython",
                            "Points",
                            transect_id + "_Points.json",
                        ),
                        "r",
                    )
                    transect_points = json.load(f)
                    f.close()
                    transect_points_batch.append(transect_points)
                transect_feat_batch = ee.FeatureCollection(
                    [item for sublist in transect_points_batch for item in sublist]
                )

                # create composite images
                imgcolfilter = (
                    ee.ImageCollection(collection_name[collection])
                    .filterDate(tempsdate, tempedate)
                    .filterBounds(transect_feat_batch)
                )
                imgcolbands = imgcolfilter.select(
                    band_names[collection], band_std_names
                )
                imgcol = imgcolbands.filterMetadata(
                    "CLOUDY_PIXEL_PERCENTAGE", "less_than", cloudcover_limit
                )
                imagecol2 = imgcol.map(bicubic)
                image = imagecol2.reduce(
                    ee.Reducer.intervalMean(14, 15)
                )  # reduce image to interval values
                image = image.select(band_std_names_mean, band_std_names).clip(
                    transect_feat_batch
                )
                ndwi = image.normalizedDifference(["Green", "NIR"]).select(
                    ["nd"], ["NDWI"]
                )
                ndvi = image.normalizedDifference(["NIR", "Red"]).select(
                    ["nd"], ["NDVI"]
                )
                imgind = image.addBands(ndwi).addBands(ndvi)

                # classify image
                prediction = imgind.classify(classifier_image)

                class_sample = prediction.unmask(-999, False).sampleRegions(
                    transect_feat_batch, ["class"], scale, tileScale=4
                )
                class_sample_info_T = class_sample.aggregate_array(
                    "classification"
                ).getInfo()

                for t in range(len(transect_points_batch)):
                    class_sample_info = pd.DataFrame(
                        {"class": class_sample_info_T[t * 151 : t * 151 + 151]}
                    )

                    class_count = class_sample_info["class"].value_counts()
                    if class_sample_info["class"].isin([1]).any().any():
                        transect_df.loc[r + t, "sand"] = (
                            class_count[1] * dist_steps
                        )  # transect_df["sand"][r + t] = class_count[1] * dist_steps
                    else:
                        transect_df.loc[r + t, "sand"] = (
                            0  # transect_df["sand"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([2]).any().any():
                        transect_df.loc[r + t, "mud"] = (
                            class_count[2] * dist_steps
                        )  # transect_df["mud"][r + t] = class_count[2] * dist_steps
                    else:
                        transect_df.loc[r + t, "mud"] = (
                            0  # transect_df["mud"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([3]).any().any():
                        transect_df.loc[r + t, "water"] = (
                            class_count[3] * dist_steps
                        )  # transect_df["water"][r + t] = class_count[3] * dist_steps
                    else:
                        transect_df.loc[r + t, "water"] = (
                            0  # transect_df["water"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([4]).any().any():
                        transect_df.loc[r + t, "vegetation"] = (
                            class_count[4] * dist_steps
                        )  # transect_df["vegetation"][r + t] = class_count[4] * dist_steps
                    else:
                        transect_df.loc[r + t, "vegetation"] = (
                            0  # transect_df["vegetation"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([5]).any().any():
                        transect_df.loc[r + t, "other"] = (
                            class_count[5] * dist_steps
                        )  # transect_df["other"][r + t] = class_count[5] * dist_steps
                    else:
                        transect_df.loc[r + t, "other"] = (
                            0  # transect_df["other"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([6]).any().any():
                        transect_df.loc[r + t, "turbid"] = (
                            class_count[6] * dist_steps
                        )  # transect_df["turbid"][r + t] = class_count[6] * dist_steps
                    else:
                        transect_df.loc[r + t, "turbid"] = (
                            0  # transect_df["turbid"][r + t] = 0
                        )
                    if class_sample_info["class"].isin([7]).any().any():
                        transect_df.loc[r + t, "dry"] = (
                            class_count[7] * dist_steps
                        )  # transect_df["dry"][r + t] = class_count[7] * dist_steps
                    else:
                        transect_df.loc[r + t, "dry"] = (
                            0  # transect_df["dry"][r + t] = 0
                        )

                    # sample along transect for slope

                    # get information in correct formats
                    class_profile = class_sample_info
                    dem_profile = pd.DataFrame(
                        transect_df["dem profile"][r + t], columns=["dem"]
                    )
                    dem_profile = dem_profile.replace({np.nan: 0})

                    # find land-sea point (approx. MSL = 0 m elevation, SDS point is too noisy / variable to use in this analysis + often without elevation + other transect length SMonitor & Romy analysis)
                    elev_gz = [
                        idx
                        for idx in range(len(dem_profile) - 1)
                        if dem_profile["dem"][idx] > 0
                    ]  # neglects 0 or negative elevations (bathy, which would be artifacts)
                    if len(elev_gz) > 0:
                        land_sea_idx = np.max(elev_gz)  # max idx with still elevation
                        # land_sea_elev = dem_profile["dem"][land_sea_idx]
                        class_profile_red = class_profile.iloc[
                            0 : land_sea_idx + 1
                        ]  # reduce class profile from 0 (land) to land-sea point

                        # find the start and end indices of the vegetation and mud class, determine the elevations and calculate the slopes
                        veg_indices = class_profile_red.index[
                            (class_profile_red["class"] == 4)
                            | (class_profile_red["class"] == 7)
                        ].tolist()  # assumes unbroken blob
                        mud_indices = class_profile_red.index[
                            class_profile_red["class"] == 2
                        ].tolist()  # assumes unbroken blob
                        if (
                            len(veg_indices) > 1
                        ):  # if no or only one veg point, slope cannot be calculated
                            veg_start_idx = veg_indices[0]
                            veg_start_elv = dem_profile["dem"][
                                veg_start_idx
                            ]  # most landward veg point
                            veg_end_idx = veg_indices[-1]
                            veg_end_elv = dem_profile["dem"][
                                veg_end_idx
                            ]  # most seaward veg point
                            # slope
                            veg_slope = abs(
                                (veg_start_elv - veg_end_elv)
                                / ((veg_end_idx - veg_start_idx) * dist_steps)
                            )  # assumes always positive
                            transect_df.loc[r + t, "veg_slope"] = (
                                veg_slope  # transect_df["veg_slope"][r + t] = veg_slope
                            )

                        if (
                            len(mud_indices) > 1
                        ):  # if no or only one mud point, slope cannot be calculated
                            mud_start_idx = mud_indices[0]
                            mud_start_elv = dem_profile["dem"][
                                mud_start_idx
                            ]  # most landward mud point
                            mud_end_idx = mud_indices[-1]
                            mud_end_elv = dem_profile["dem"][
                                mud_end_idx
                            ]  # most seaward mud point
                            # slope
                            mud_slope = abs(
                                (mud_start_elv - mud_end_elv)
                                / ((mud_end_idx - mud_start_idx) * dist_steps)
                            )
                            transect_df.loc[r + t, "mud_slope"] = (
                                mud_slope  # transect_df["mud_slope"][r + t] = mud_slope
                            )

            ### end batch
            transect_df.drop(["dem profile"], axis=1, inplace=True)

            ### classify transects per box
            td_tot = transect_df.loc[:, features]
            if not td_tot.isnull().values.any():
                td_norm = (td_tot - tt_f.min()) / (tt_f.max() - tt_f.min())
                td_pred = rfc_hybrid.predict(td_norm.loc[:, features])
                transect_df["prediction"] = td_pred

            time.sleep(1)
            end = time.time()
            print(str(round((end - begin), 1)) + " sec")

            # export classified transects per box
            featuresjson = []
            for n in range(len(transect_df)):
                line = LineString(
                    [
                        (
                            transect_df["New_Start_lon"][n],
                            transect_df["New_Start_lat"][n],
                        ),
                        (transect_df["New_End_lon"][n], transect_df["New_End_lat"][n]),
                    ]
                )
                featuresjson.append(
                    Feature(
                        geometry=line,
                        properties={
                            "transect_id": str(transect_df["transect_id"][n]),
                            "box_id": str(transect_df["box_id"][n]),
                            "country_name": str(transect_df["country_name"][n]),
                            "continent": str(transect_df["continent"][n]),
                            "prediction": str(transect_df["prediction"][n]),
                            "flag_sandy": str(transect_df["flag_sandy"][n]),
                            "abs lat": str(transect_df["abs lat"][n]),
                            "latitude": str(transect_df["Center_lat"][n]),
                            "sand": str(transect_df["sand"][n]),
                            "mud": str(transect_df["mud"][n]),
                            "water": str(transect_df["water"][n]),
                            "vegetation": str(transect_df["vegetation"][n]),
                            "other": str(transect_df["other"][n]),
                            "turbid": str(transect_df["turbid"][n]),
                            "dry": str(transect_df["dry"][n]),
                            "height var": str(transect_df["height var"][n]),
                            "height max": str(transect_df["height max"][n]),
                            "mangrove": str(transect_df["mangrove"][n]),
                            "intertidal": str(transect_df["intertidal"][n]),
                            "gsw": str(transect_df["gsw"][n]),
                            "maxtemp": str(transect_df["maxtemp"][n]),
                            "mintemp": str(transect_df["mintemp"][n]),
                            "tidal range": str(transect_df["tidal range"][n]),
                            "veg_slope": str(transect_df["veg_slope"][n]),
                            "mud_slope": str(transect_df["mud_slope"][n]),
                        },
                    )
                )

            feature_collection = FeatureCollection(featuresjson)
            geojsonname = "\\" + box_shuffle[0][box] + "_ClassifiedTransects2020.json"
            filepath = os.path.join(
                results, scriptname, box_shuffle[0][box][0:7], "JSONperBOX"
            )
            with open(filepath + geojsonname, "w") as f:
                dump(feature_collection, f)
            f.close()

            filename = "\\" + box_shuffle[0][box] + "_ClassifiedTransects2020.csv"
            filepath = os.path.join(
                results, scriptname, box_shuffle[0][box][0:7], "CSVperBOX"
            )
            transect_df.to_csv(filepath + filename)

        except Exception as e:
            print(e)
