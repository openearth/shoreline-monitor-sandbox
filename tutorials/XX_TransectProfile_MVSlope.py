# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:27:46 2022

@author: hulskamp
"""

import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import datetime
import os.path
import json
import time
import pandas as pd
import os
import time
import math
import csv
import statistics
import sklearn
from sklearn.ensemble import RandomForestClassifier
from geojson import MultiPoint, Feature, FeatureCollection, dump, LineString
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None
import ee

ee.Initialize()

# get filename
folder = (
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Scripts"
)
results = (
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Results"
)
scriptname = "TransectProfile"
# outputfolder
os.chdir(folder)
if not os.path.exists(os.path.join(results, scriptname)):
    os.mkdir(os.path.join(results, scriptname))

# %%
# define collection and bands
collection = "IM_S2"
collection_name = {"IM_S2": "COPERNICUS/S2"}
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
    r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Paper Final Scripts and Data\Data\GlobalTransectsForClassification.csv",
    delimiter=",",
)
global_transects = global_transects.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

# Add box_ids
boxes = []
for b in range(len(global_transects)):
    box_id = global_transects["transect_id"][b][0:11]
    boxes.append(box_id)
global_transects["box_id"] = boxes

global_transects[
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
    ]
] = None

global_transects["abs lat"] = abs(global_transects["Center_lat"])

# box_unique = pd.DataFrame(np.unique(boxes))
# box_shuffle = box_unique.sample(frac = 1, random_state = 1).reset_index(drop=True) # shuffled order

# choosebox = ['BOX_117_039'] # suriname: transects set t in line 151 -> range(0,5)
choosebox = ["BOX_079_005"]  # madagascar: transects set t in line 151 -> range(20,25)
box_shuffle = pd.DataFrame(choosebox)


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
        ]
    ] = None
    transect_df["abs lat"] = abs(transect_df["Center_lat"])

    print(box, box_shuffle[0][box], len(transect_df), "transects")

    for t in range(20, 25):  # len(transect_df)):
        # load transect
        transect_id = transect_df["transect_id"][t]
        f = open(
            os.path.join(
                r"P:\1000545-054-globalbeaches\07_Muddy_Coasts\Mud_RHulskamp\muddy results\2021-09-23_CreateTransects_withPython\Points",
                transect_id + "_Points.json",
            ),
            "r",
        )
        transect_points = json.load(f)
        f.close()
        transect_feat = ee.FeatureCollection(transect_points)

        # get coordinates and center of transects
        coordinates = [
            [
                [transect_df["New_Start_lon"][t], transect_df["New_Start_lat"][t]],
                [transect_df["New_End_lon"][t], transect_df["New_End_lat"][t]],
            ]
        ]
        center = ee.Geometry.Point(
            [transect_df["Center_lon"][t], transect_df["Center_lat"][t]]
        )
        multiline = ee.Geometry.MultiLineString(coordinates)
        buffer = multiline.buffer(25)
        bufferinfo = buffer.getInfo()
        buffer_bounds_trans = bufferinfo["coordinates"][0]
        aoi_trans = ee.FeatureCollection(ee.Geometry.Polygon(buffer_bounds_trans))

        # create composite images
        imgcolfilter = (
            ee.ImageCollection(collection_name[collection])
            .filterDate(tempsdate, tempedate)
            .filterBounds(aoi_trans)
        )
        imgcolbands = imgcolfilter.select(band_names[collection], band_std_names)
        imgcol = imgcolbands.filterMetadata(
            "CLOUDY_PIXEL_PERCENTAGE", "less_than", cloudcover_limit
        )

        imagecol2 = imgcol.map(bicubic)
        image = imagecol2.reduce(
            ee.Reducer.intervalMean(14, 15)
        )  # reduce image to interval values
        image = image.select(band_std_names_mean, band_std_names).clip(aoi_trans)
        ndwi = image.normalizedDifference(["Green", "NIR"]).select(["nd"], ["NDWI"])
        ndvi = image.normalizedDifference(["NIR", "Red"]).select(["nd"], ["NDVI"])
        imgind = image.addBands(ndwi).addBands(ndvi)

        # classify image
        prediction = imgind.classify(classifier_image)

        # sample over transect
        # print('Class')
        class_sample = prediction.unmask(-999, False).sampleRegions(
            transect_feat, ["class"], scale, tileScale=4
        )
        class_sample_info = pd.DataFrame(
            {"class": class_sample.aggregate_array("classification").getInfo()}
        )
        class_count = class_sample_info["class"].value_counts()
        if class_sample_info["class"].isin([1]).any().any():
            transect_df["sand"][t] = class_count[1] * dist_steps
        else:
            transect_df["sand"][t] = 0
        if class_sample_info["class"].isin([2]).any().any():
            transect_df["mud"][t] = class_count[2] * dist_steps
        else:
            transect_df["mud"][t] = 0
        if class_sample_info["class"].isin([3]).any().any():
            transect_df["water"][t] = class_count[3] * dist_steps
        else:
            transect_df["water"][t] = 0
        if class_sample_info["class"].isin([4]).any().any():
            transect_df["vegetation"][t] = class_count[4] * dist_steps
        else:
            transect_df["vegetation"][t] = 0
        if class_sample_info["class"].isin([5]).any().any():
            transect_df["other"][t] = class_count[5] * dist_steps
        else:
            transect_df["other"][t] = 0
        if class_sample_info["class"].isin([6]).any().any():
            transect_df["turbid"][t] = class_count[6] * dist_steps
        else:
            transect_df["turbid"][t] = 0
        if class_sample_info["class"].isin([7]).any().any():
            transect_df["dry"][t] = class_count[7] * dist_steps
        else:
            transect_df["dry"][t] = 0

        # print('DEM')
        dem = ee.Image("MERIT/DEM/v1_0_3")
        dem_sample = dem.unmask(-999, False).sampleRegions(transect_feat, ["id"], scale)
        dem_sample_info = pd.DataFrame(
            {"dem": dem_sample.aggregate_array("dem").getInfo()}
        ).replace({-999: np.NaN})
        transect_df["height max"][t] = np.max(dem_sample_info["dem"])
        transect_df["height var"][t] = np.var(dem_sample_info["dem"])

        # print('Mangrove occurrence')
        mangrove = ee.ImageCollection("LANDSAT/MANGROVE_FORESTS").first()
        mangrove_sample = mangrove.unmask(-999, False).sampleRegions(
            transect_feat, ["id"], scale
        )
        mangrove_sample_info = pd.DataFrame(
            {"mangrove": mangrove_sample.aggregate_array("1").getInfo()}
        )
        mangrove_count = mangrove_sample_info[
            "mangrove"
        ].value_counts()  # if one point contains mangrove (1), then mangroves occur
        if mangrove_sample_info["mangrove"].isin([1]).any().any():
            transect_df["mangrove"][t] = 1
        else:
            transect_df["mangrove"][t] = 0

        # print('Intertidal area')
        intertidal = ee.ImageCollection(
            "UQ/murray/Intertidal/v1_1/global_intertidal"
        ).first()
        intertidal_sample = intertidal.unmask(-999, False).sampleRegions(
            transect_feat, ["id"], scale
        )
        intertidal_sample_info = pd.DataFrame(
            {
                "intertidal": intertidal_sample.aggregate_array(
                    "classification"
                ).getInfo()
            }
        )
        intertidal_count = intertidal_sample_info[
            "intertidal"
        ].value_counts()  # count points that contain intertidal flat (1)
        if intertidal_sample_info["intertidal"].isin([1]).any().any():
            transect_df["intertidal"][t] = intertidal_count[1] * dist_steps
        else:
            transect_df["intertidal"][t] = 0

        # print('Global Surface Water')
        GSW = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select("occurrence")
        GSW_sample = GSW.unmask(-999, False).sampleRegions(transect_feat, ["id"], scale)
        GSW_sample_info = pd.DataFrame(
            {"gsw": GSW_sample.aggregate_array("occurrence").getInfo()}
        )
        gsw90 = []
        for g in range(len(GSW_sample_info["gsw"])):
            if GSW_sample_info["gsw"][g] > 5 and GSW_sample_info["gsw"][g] < 95:
                gsw90value = GSW_sample_info["gsw"][g]
                gsw90.append(gsw90value)
        transect_df["gsw"][t] = len(gsw90) * dist_steps

        # print('Temperature')
        temp = ee.Image("WORLDCLIM/V1/BIO")
        pts = ee.FeatureCollection(center).map(
            lambda i: get_nearest_value(i, temp.select("bio05"))
        )
        maxtemp_mean = np.array(pts.aggregate_array("bio05").getInfo()) * 0.1
        pts = ee.FeatureCollection(center).map(
            lambda i: get_nearest_value(i, temp.select("bio06"))
        )
        mintemp_mean = np.array(pts.aggregate_array("bio06").getInfo()) * 0.1
        transect_df["maxtemp"][t] = maxtemp_mean[0]
        transect_df["mintemp"][t] = mintemp_mean[0]

        # print('Tidal range')
        tidal = ee.Image("projects/dgds-gee/gtsm/tidal_indicators")
        pts = ee.FeatureCollection(center).map(
            lambda i: get_nearest_value(i, tidal.select("mean_higher_high_water"))
        )
        mhhw_mean = pts.aggregate_array(
            "mean_higher_high_water"
        ).getInfo()  # generates a list of all point values, at the moment is only 1 value, but could be expanded easily to get centers as a feature collection
        pts = ee.FeatureCollection(center).map(
            lambda i: get_nearest_value(i, tidal.select("mean_lower_low_water"))
        )
        mllw_mean = pts.aggregate_array("mean_lower_low_water").getInfo()

        transect_df["mhhw"][t] = mhhw_mean[0]
        transect_df["mllw"][t] = mllw_mean[0]
        transect_df["tidal range"][t] = transect_df["mhhw"][t] - transect_df["mllw"][t]

        class_profile = class_sample_info
        dem_profile = dem_sample_info.replace({np.NaN: 0})
        newdist = list(np.arange(0, len(transect_points) * dist_steps, dist_steps))
        gsw_profile = GSW_sample_info.replace({-999: 0})
        gsw_list = [
            i
            for i in range(len(gsw_profile))
            if gsw_profile["gsw"][i] > 4 and gsw_profile["gsw"][i] < 96
        ]
        intertidal_profile = intertidal_sample_info
        intertidal_list = [
            i
            for i in range(len(intertidal_profile))
            if intertidal_profile["intertidal"][i] == 1
        ]
        mangrove_profile = mangrove_sample_info
        mangrove_list = [
            i
            for i in range(len(mangrove_profile))
            if mangrove_profile["mangrove"][i] == 1
        ]

        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.plot(newdist, dem_profile, label="Elevation profile", zorder=3, linewidth=2)
        ax.axhspan(mllw_mean[0], mhhw_mean[0], alpha=0.2, zorder=0, label="Tidal range")
        ax.hlines(
            0,
            xmin=0,
            xmax=1500,
            linewidth=1,
            linestyles="dashed",
            color="red",
            zorder=3,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 1] * 0,
            color="#ffe30f",
            label="Sand",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 2] * 0,
            color="#6d2c0c",
            label="Mud",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 3] * 0,
            color="#3581eb",
            label="Water",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 6] * 0,
            color="#0000ff",
            label="Turbid water",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 4] * 0,
            color="#35b614",
            label="Vegetation",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 7] * 0,
            color="#4f6701",
            label="Dry vegetation",
            zorder=2,
        )
        ax.scatter(
            newdist,
            class_profile[class_profile == 5] * 0,
            color="#ff0101",
            label="Other",
            zorder=2,
        )
        ax.set_xlabel("Distance [m]", fontsize=22)
        ax.set_ylabel("Elevation [m]", color="tab:blue", fontsize=22)
        # ax.set_ylim(-1, 1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
        title = (
            "Transect profile " + transect_id + ", " + transect_df["country_name"][t]
        )
        # ax.set_title(title, fontsize=20)
        ax2 = ax.twinx()
        ax2.plot(
            newdist,
            gsw_profile,
            color="goldenrod",
            label="Water probability",
            zorder=3,
            linewidth=2,
        )
        ax2.vlines(
            np.multiply(gsw_list, 10),
            ymin=0,
            ymax=100,
            linewidth=4,
            color="goldenrod",
            zorder=1,
            alpha=0.2,
            label="Transition zone",
        )
        ax2.set_ylabel("Water probability [%]", color="goldenrod", fontsize=22)
        plt.yticks(fontsize=16)
        ax2.vlines(
            np.multiply(intertidal_list, 10),
            ymin=0,
            ymax=100,
            linewidth=4,
            color="darkviolet",
            zorder=1,
            alpha=0.2,
            label="Tidal flat",
        )
        ax2.vlines(
            np.multiply(mangrove_list, 10),
            ymin=0,
            ymax=100,
            linewidth=4,
            color="darkgreen",
            zorder=1,
            alpha=0.2,
            label="Mangrove",
        )
        # fig.legend(loc = 'center right', bbox_to_anchor=(1.2,0.728), fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handleso = [
            handles[2],
            handles[0],
            handles[3],
            handles[1],
            handles[4],
            handles2[0],
            handles[5],
            handles2[1],
            handles[6],
            handles[7],
            handles[8],
        ]
        labelso = [
            labels[2],
            labels[0],
            labels[3],
            labels[1],
            labels[4],
            labels2[0],
            labels[5],
            labels2[1],
            labels[6],
            labels[7],
            labels[8],
        ]
        # fig.legend(handleso,labelso,loc = 'center right', bbox_to_anchor=(1.5,0.728), fontsize=16, ncol = 3)
        # textstr = '\n'.join((
        #                     r'$\mathrm{Lat \ [\degree]}=%.2f$' % (transect_df['Center_lat'][t], ),
        #                     r'$\mathrm{T_{max} \ [\degree C]}=%.2f$' % (maxtemp_mean[0], ),
        #                     r'$\mathrm{T_{min} \ [\degree C]}=%.2f$' % (mintemp_mean[0], ),
        #                     r'$\mathrm{TR \ [m]}=%.2f$' % (mhhw_mean[0]-mllw_mean[0], )))
        # props = dict(boxstyle='square',facecolor='white', alpha=0.2)
        # ax.text(1.083, 0.45, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='bottom', bbox = props)
        plt.savefig(
            os.path.join(results, scriptname, title), dpi=300, bbox_inches="tight"
        )
        plt.show()

        # plt.figure(figsize=(12,6.75))
        # plt.legend(handleso,labelso,fontsize=16,ncol = 7,bbox_to_anchor =(1, 1.3))
        # plt.show()

        # MSIC transect
        filename = (
            "Transect profile " + transect_id + ", " + transect_df["country_name"][t]
        )
        scale = 10
        ee.batch.Export.image.toDrive(
            image=prediction,
            description=filename,
            folder=scriptname,
            scale=scale,
            crs="EPSG:3857",
            region=buffer_bounds_trans,
            maxPixels=1e10,
        ).start()
