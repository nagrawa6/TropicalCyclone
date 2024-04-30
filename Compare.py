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

from geopy.distance import geodesic
from haversine import haversine, Unit
import math
import pandas as pd
import matplotlib.pyplot as plt

from geopy.distance import geodesic
from haversine import haversine, Unit
import math

def calculate_angle(lat1, lon1, lat2, lon2, lat3, lon3):
    # Calculate the distances between the points
    d1 = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
    d2 = haversine((lat2, lon2), (lat3, lon3), unit=Unit.KILOMETERS)
    d3 = haversine((lat1, lon1), (lat3, lon3), unit=Unit.KILOMETERS)
    
    # Check if any distance is zero
    if d1 == 0 or d2 == 0 or d3 == 0:
        return 0

    # Determine the orientation of the points using cross product
    cross_product = (lon2 - lon1) * (lat3 - lat1) - (lat2 - lat1) * (lon3 - lon1)
        
    # Calculate the angle between the lines using the Law of Cosines
    try:
        angle = math.acos((d1 ** 2 + d2 ** 2 - d3 ** 2) / (2 * d1 * d2))
    except ValueError:
        return 0
    
    # Adjust the sign of the angle based on the orientation of the points
    if cross_product > 0:
        angle = 2 * math.pi - angle
        
    angle_deg = math.degrees(angle)

    return angle_deg

def calculate_ate(lat1, lon1, lat2, lon2, lat3, lon3):
    angle = calculate_angle(lat1, lon1, lat2, lon2, lat3, lon3)
    return (haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS) * 
            math.cos(math.radians(angle)))

def calculate_cte(lat1, lon1, lat2, lon2, lat3, lon3):
    angle = calculate_angle(lat1, lon1, lat2, lon2, lat3, lon3)
    return (haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS) * 
            math.sin(math.radians(angle)))

def calculate_tte(lat1, lon1, lat2, lon2, lat3, lon3):
    return (haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS))

def create_best_dataframe(year, df_best1, x):
    a = df_best1.loc[df_best1.index.get_level_values('date').year == year]
    df_best1 = a
    df_best2 = df_best1.shift(-1)
    df = pd.merge(df_best1, df_best2, left_index=True, right_index=True)
    x = pd.concat([x, df])
    return x

def create_best_matrix(df_best1):
    x = pd.DataFrame()
    for year in df_best1.index.get_level_values('date').year.unique():
        x = create_best_dataframe(year, df_best1, x)
    selected_columns = [col for col in x.columns if 'lat' in col or 'lon' in col or 'CY_x' in col]

    df_selected = x[selected_columns].dropna()
    
    # Create the new row
    new_index = df_selected.index[-1][0] + timedelta(hours=6), df_selected.index[-1][1]
    new_row_data = df_selected.loc[:, ['blat_y', 'blon_y', 'blat_x', 'blon_x']].iloc[-1, :]

    # Create a new DataFrame for the new row
    new_row_df = pd.DataFrame([new_row_data.values], columns=['blat_x', 'blon_x', 'blat_y', 'blon_y'], index=[new_index])

    # Append the new row to the DataFrame 'a'
    df_selected = df_selected.append(new_row_df)

    return df_selected

def calculate_error(b):
    x = create_best_matrix(b.loc[:, ['blat', 'blon']])
    
    # Merge the two DataFrames based on the new index column
    merged_df = pd.merge(b, x, right_on=x.index, left_on=b.index, how='inner')

    merged_df['key_0'] = [str(value) for value in merged_df['key_0']]

    # Splitting the 'coordinates' column into two separate columns
    merged_df[['date', 'CY']] = merged_df['key_0'].str.strip('()').str.split(',', expand=True)

    # Dropping the original 'coordinates' column
    merged_df = merged_df.drop('key_0', axis=1)

    # Setting 'x' and 'y' columns as index columns
    merged_df.set_index(['date', 'CY'], inplace=True)
    
    # Apply calculate_ate function element-wise to create ATE column
    merged_df.loc[:, 'mod_ATE'] = merged_df.apply(lambda row: calculate_ate(row['plat'], row['plon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)
    
    # Apply calculate_ate function element-wise to create ATE column
    merged_df.loc[:, 'ens_ATE'] = merged_df.apply(lambda row: calculate_ate(row['mlat'], row['mlon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)

    # Apply calculate_cte function element-wise to create CTE column
    merged_df.loc[:, 'mod_CTE'] = merged_df.apply(lambda row: calculate_cte(row['plat'], row['plon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)

    # Apply calculate_cte function element-wise to create CTE column
    merged_df.loc[:, 'ens_CTE'] = merged_df.apply(lambda row: calculate_cte(row['mlat'], row['mlon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)
    
    # Apply calculate_tte function element-wise to create TTE column
    merged_df.loc[:, 'mod_TTE'] = merged_df.apply(lambda row: calculate_tte(row['plat'], row['plon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)

    # Apply calculate_tte function element-wise to create TTE column
    merged_df.loc[:, 'ens_TTE'] = merged_df.apply(lambda row: calculate_tte(row['mlat'], row['mlon'], row['blat_x'], row['blon_x'], row['blat_y'], row['blon_y']), axis=1)
    
    last_row = merged_df.iloc[-1]  # Get the last row
    last_row[-6:-2] *= -1   
    merged_df.iloc[-1] = last_row
    
    
    # Calculate the correlation between the two columns
    print("Correlation between mod_ATE and mod_TTE:", merged_df['mod_ATE'].corr(merged_df['mod_TTE']))
    print("Correlation between mod_CTE and mod_TTE:", merged_df['mod_CTE'].corr(merged_df['mod_TTE']))
    print("Correlation between ens_ATE and ens_TTE:", merged_df['ens_ATE'].corr(merged_df['ens_TTE']))
    print("Correlation between ens_CTE and ens_TTE:", merged_df['ens_CTE'].corr(merged_df['ens_TTE']))
    
    return merged_df

def tte_graph(b, fig_size=(8, 6)):    
    x_values = [6*i for i in range(len(b))]
    
    plt.figure(figsize=fig_size)

    # Plotting
    plt.plot(x_values, b['ens_TTE'], label='GEFS Mean Error', color='r')  # Red line
    plt.plot(x_values, b['mod_TTE'], label='CNN Model Error', color='b')  # Blue line

    # Adding labels and title
    plt.xlabel('Forecast lead time (hours)')
    plt.ylabel('Total Track Error (km)')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()

def ate_graph(b, fig_size=(8, 6)):
    x_values = [6*i for i in range(len(b))]
    
    plt.figure(figsize=fig_size)
    
    b['ens_ATE'] = b['ens_ATE'].abs()
    b['mod_ATE'] = b['mod_ATE'].abs()

    # Plotting
    plt.plot(x_values, b['ens_ATE'], label='GEFS Mean Error', color='r')  # Red line
    plt.plot(x_values, b['mod_ATE'], label='CNN Model Error', color='b')  # Blue line

    # Adding labels and title
    plt.xlabel('Forecast lead time (hours)')
    plt.ylabel('Along Track Error (km)')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()
    
def cte_graph(b, fig_size=(8, 6)):
    x_values = [6*i for i in range(len(b))]
    
    b['ens_CTE'] = b['ens_CTE'].abs()
    b['mod_CTE'] = b['mod_CTE'].abs()
    
    plt.figure(figsize=fig_size)

    # Plotting
    plt.plot(x_values, b['ens_CTE'], label='GEFS Mean Error', color='r')  # Red line
    plt.plot(x_values, b['mod_CTE'], label='CNN Model Error', color='b')  # Blue line

    # Adding labels and title
    plt.xlabel('Forecast lead time (hours)')
    plt.ylabel('Cross Track Error (km)')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()
    
def all_on_same(b, fig_size=(8, 6)):
    b['ens_ATE'] = b['ens_ATE'].abs()
    b['mod_ATE'] = b['mod_ATE'].abs()
    
    b['ens_CTE'] = b['ens_CTE'].abs()
    b['mod_CTE'] = b['mod_CTE'].abs()

    x_values = [6*i for i in range(len(b))]
    
    plt.figure(figsize=fig_size)

    # Plotting TTE
    plt.plot(x_values, b['ens_TTE'], label='GEFS Mean TTE', color='r', linestyle='-')  # Red solid line
    plt.plot(x_values, b['mod_TTE'], label='CNN Model TTE', color='b', linestyle='-')  # Blue solid line

    # Plotting ATE
    plt.plot(x_values, b['ens_ATE'], label='GEFS Mean ATE', color='magenta', linestyle=':')  # Red dashed line
    plt.plot(x_values, b['mod_ATE'], label='CNN Model ATE', color='teal', linestyle=':')  # Blue dashed line

    # Plotting CTE
    plt.plot(x_values, b['ens_CTE'], label='GEFS Mean CTE', color='crimson', linestyle='--')  # Red dash-dot line
    plt.plot(x_values, b['mod_CTE'], label='CNN Model CTE', color='darkblue', linestyle='--')  # Blue dash-dot line

    # Adding labels and title
    plt.xlabel('Forecast lead time (hours)')
    plt.ylabel('Error (km)')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()
    
def error_graph(b, fig_size=(8, 6)):
    b = calculate_error(b)
    '''
    all_on_same(b, fig_size)
    '''
    tte_graph(b, fig_size)
    ate_graph(b, fig_size)
    cte_graph(b, fig_size)
    
    
    
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