import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import gzip
import shutil
import requests
import numpy as np
from datetime import datetime, timedelta

import StormData

from bs4 import BeautifulSoup

import Dataloader as DL
import ReAnalysisData as rad
import ensemble_pn
import model
#import model1
import model2
import model3

import Lmodel as model1
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

def create_dataframe(start_num, test, outputs, NUM_OF_INTERVALS):
    test1 = test.reset_index()
    a = test1[['date', 'CY']]

    model1_output = outputs[start_num:start_num+1, :]
    reshaped_array = model1_output.reshape(-1, 2)

    df1 = pd.DataFrame(data = reshaped_array, columns = ['lat', 'lon'])
    df1['date'] = [a['date'][start_num] + timedelta(hours=6 * i) for i in range(len(df1))]
    df1['CY'] = [a['CY'][start_num] for i in range(len(df1))]

    df1.set_index(['date','CY'], inplace = True)
    df1b = df1
    len(df1b)

    _df_mean = test.iloc[start_num:start_num+1, -NUM_OF_INTERVALS*4-1:-NUM_OF_INTERVALS*2-1].to_numpy()
    reshaped_mean = np.reshape(_df_mean, (2, NUM_OF_INTERVALS))

    df2 = pd.DataFrame(data = reshaped_mean.T, columns = ['lat', 'lon'])
    df2['date'] = [a['date'][start_num] + timedelta(hours=6 * i) for i in range(len(df2))]
    df2['CY'] = [a['CY'][start_num] for i in range(len(df2))]
    df2.set_index(['date','CY'], inplace = True)
    _df_mean = df2
    len(_df_mean)

    _df_best = test.iloc[start_num:start_num+1, -NUM_OF_INTERVALS*2-1:-1].to_numpy()
    reshaped_best = _df_best.reshape(-1, 2)

    df3 = pd.DataFrame(data = reshaped_best, columns = ['lat', 'lon'])
    df3['date'] = [a['date'][start_num] + timedelta(hours=6 * i) for i in range(len(df3))]
    df3['CY'] = [a['CY'][start_num] for i in range(len(df3))]

    df3.set_index(['date','CY'], inplace = True)
    _dfb = df3
    len(_dfb)

    a = pd.concat([_df_mean, _dfb], keys=['date', 'CY'], axis=1)
    a = a.droplevel(0, axis = 1)
    a.columns = ['mlat', 'mlon', 'blat', 'blon']
    a = pd.concat([a, df1b], keys=['date', 'CY'], axis=1)
    a = a.droplevel(0, axis = 1)
    a.columns = ['mlat', 'mlon', 'blat', 'blon', 'plat', 'plon']
    a = a.dropna(axis=0, thresh=6)
    
    return a

def plot_compare(a, extent, position):
    # Dictionary to map DataFrames to colors
    colors = ['red', 'green', 'blue']

    # Create the map figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add lat/lon gridlines
    ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # Plot each DataFrame on the map and collect coordinates
    all_coordinates = []
    lon_cols = [col for col in a.columns if 'lon' in col]
    lat_cols = [col for col in a.columns if 'lat' in col]

    # If the number of DataFrames exceeds the number of colors, repeat the colors
    num_colors_needed = len(lon_cols)
    colors = colors * ((num_colors_needed // len(colors)) + 1)

    index = 0  # Reset index counter for each DataFrame
    for lon_col, lat_col in zip(lon_cols, lat_cols):
        ax.plot(a[lon_col], a[lat_col], 'o', markersize=4, color=colors[index], transform=ccrs.Geodetic())
        coordinates = list(zip(a[lon_col], a[lat_col]))
        all_coordinates.extend(coordinates)

        # Annotate each point with an index starting from 1 for each DataFrame
        index_counter = 0
        for lon, lat in coordinates:
            ax.annotate(str(index_counter*6), xy=(lon, lat), xytext=(2, 2), textcoords="offset points", fontsize=8, color='black')
            index_counter += 1
        index += 1

    # Set the map extent to show all points
    min_lon = min(lon for lon, lat in all_coordinates)
    max_lon = max(lon for lon, lat in all_coordinates)
    min_lat = min(lat for lon, lat in all_coordinates)
    max_lat = max(lat for lon, lat in all_coordinates)
    ax.set_extent([min_lon - extent, max_lon + extent, min_lat - extent, max_lat + extent], crs=ccrs.PlateCarree())

    # Add some map features
    ax.coastlines()
    ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(ccrs.cartopy.feature.OCEAN, edgecolor='none')

    # Add a legend
    labels = ['GEFS Ensemble Mean', 'Best Track', 'CNN Model Prediction']
    ax.legend(labels[:len(lon_cols)], loc=position, fontsize=10)

    # Show the plot
    plt.show()

# Function to calculate the total absolute differences between DataFrames and return the closest overall
def compare_closeness(start_num, test, outputs, NUM_OF_INTERVALS, _print = False):
    
    a = create_dataframe(start_num, test, outputs, NUM_OF_INTERVALS)

    dfm, dfb, dfp = a[['mlat', 'mlon']], a[['blat', 'blon']], a[['plat', 'plon']]
    
    # Calculate RMSE
    _mean_rmse = np.sqrt(((dfb.values - dfm.values) ** 2).mean().mean())
    _model1_rmse = np.sqrt(((dfb.values - dfp.values) ** 2).mean().mean())
    
    if _print:
        # Print Differences
        print(f"Mean Track RMSE: {_mean_rmse}")
        print(f"Model 1 Track RMSE: {_model1_rmse}")

        # Calculate Differences
        _mean_diff = abs(dfb.to_numpy() - dfm.to_numpy()).sum().sum()
        _model1_diff = abs(dfb.to_numpy() - dfp.to_numpy()).sum().sum()

        # Print Differences
        #print(f"Mean Track Diff: {_mean_diff}")
        #print(f"Model 1 Track Diff: {_model1_diff}")

        print(f"Storm Date: {a.index[0]}")
        print(f"Improvement: {(_mean_rmse - _model1_rmse)/_mean_rmse}")
    
    return ((_mean_rmse - _model1_rmse)/_mean_rmse)