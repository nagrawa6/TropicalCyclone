# Improving Tropical Cyclone Track Forecast

## Overview
Over the period from 2008 to 2022, the United States experienced a range of tropical cyclones, including hurricanes and tropical storms. The ability to predict the track of tropical cyclones is important for public safety and disaster preparedness, reducing the risk to human life and property. The Global Ensemble Forecast System (GEFS) is a sophisticated physics model created by the National Oceanic and Atmospheric Administration (NOAA) to make weather predictions spanning from days to weeks into the future. However, the GEFS Mean can have relatively large biases that can lead to large predicted track error. In order to correct the GEFS Mean 5-day forecasts for tropical cyclones in the Atlantic Basin, this research will develop a CNN model developed by taking into account data from 2,395 tropical cyclone 5-day forecast sequences across the United States in a 15 year timespan. 

## Data Sources
The 21 GEFS ensemble members were obtained every 6 hours out to day 5 forecast predictions from 2008-2022. The tropical cyclone Best Tracks and GEFS tropical cyclone storm forecast tracks were retrieved from the National Oceanic and Atmospheric Administration’s (NOAA) National Hurricane Center. Python version 3.9.13 on Jupyter Notebooks, a Python development environment, was used to develop Python code for this project.

## Methodology
### Processing: 
When gathering the GEFS data, some tropical cyclone ensemble member forecasts may be missing. Thus, when preliminarily screening the data, the missing tropical cyclone data was deleted. In addition, the short tropical cyclone sequences with a lifespan of less than 5 days were deleted to avoid affecting the accuracy. Therefore, of the total 263 tropical cyclone storms from 2008-2022 gathered, 139 tropical cyclones were included in this research.

### Modeling: 
The Python torch library was used to create the CNN model. The input to the modeling process was the data frame resulting from the data processing of the multiple tropical cyclone storms. To ensure randomness of tropical cyclone sequences inputted to the CNN Model, the order of tropical cyclone sequences data within the data frame was shuffled using a constant seed. K-fold cross-validation was used to determine a more reliable accuracy score using a k-value of 5. 

### Evaluation: 
To assess the accuracy of the CNN Model in comparison to the GEFS Mean, a crucial step involved employing the haversine function for distance calculations. The haversine function was used to calculate the distances in kilometers between the predicted latitude and longitude coordinates of both the CNN Model and GEFS Mean, and the corresponding Best Track latitude and longitude coordinates for each forecast lead time. The Relative Improvement Percentage is a valuable metric for assessing the enhancement achieved by the CNN Model over the GEFS Mean. By calculating the percentage improvement, one can quantitatively measure the extent to which the CNN Model has refined its predictive capabilities. A positive Relative Improvement Percentage signifies a reduction in the root mean squared error (RMSE), indicating that the CNN Model has produced predictions closer to the Best Track compared to the GEFS Mean.

### Overall Statistics: 
Based on graphing the median total track error as a function of forecast lead times for the GEFS Mean and CNN Model with 95% confidence intervals, the CNN Model generally outperforms the GEFS Mean, especially between 0 and 96 hours in advance. Based on graphing the overall percentage performance per forecast lead time, the CNN Model outperformed the GEFS Mean between 0 to 108 hours.

## Steps:
Software: Python version 3.9.13 on Jupyter Notebooks.

### Data:
The consolidated GEFS Mean tropical cyclone track prediction and Best Track data is stored in the following: df_master_X.pkl

### Model Training and Tuning: 
Taking the input data files, develop the CNN Model (or use the previously developed model: mod_five_0-1_500__3.joblib) using the following: PreProcessing v20 - five k fold.ipynb. This file performs the entire methodology for tropical cyclones from 2008 to 2022.

The tropical cyclones from 2008 to 2022 and their corresponding forecast sequences for 5-days were split into 90% training & validation (2,218 sequences) and 10% holdout evaluation datasets (177 evaluation sequences). Run analysis on the GEFS Mean and CNN Model tropical cyclone track prediction such as by comparing the Total Track Error (TTE), Along-Track Error (ATE), and Cross-Track Error (CTE). Use the TTE to compute Relative Improvement Percentage - percentage increase or decrease from the GEFS Mean TTE to the CNN Model TTE. Additionally, graph the Median Total Track Error (TTE) per forecast lead time with 95% confidence interval and Overall Percentage Performance per forecast lead time. Furthermore, plot example GEFS Mean and CNN Model  tropical cyclone track predictions compared with the Best Track on a map by calling Compare.py in the PreProcessing v20 - five k fold.ipynb. 

Additionally, test on new 2023 tropical cyclone data that the CNN Model was unaware of using the following: PreProcessing v20 - time based.ipynb. This file performs the entire methodology as above for tropical cyclones from 2023.

## Results
CNN Model achieved statistically significant lower total track error than the GEFS Mean developed by NOAA. Improvement from existing AI models developed for tropical cyclone track forecasting. The GEFS Mean has a high ATE which leads to a high TTE. This was detected by the developed CNN Model which accelerated forecasts, indicated by the Relative Improvement Percentage. When evaluated on previously unknown tropical cyclone data from 2023, the CNN Model outperformed the GEFS Mean.
