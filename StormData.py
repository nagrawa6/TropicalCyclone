import os
import pandas as pd
import gzip
import shutil
import requests
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go

#Read *.dat file and store it in dataframe
def create_year_picklefile(_year):
    path = "data/ATCF"
    print(_year)
    df_year = pd.DataFrame()
    _file_list = os.listdir(path+'/'+_year)

    # Reads raw *.dat files for a given year
    for dat_file in _file_list:
        #print(dat_file)
        with open("data/ATCF/"+_year+'/'+ dat_file) as f:
            lines = f.readlines()
            df = pd.DataFrame([sub.split(",") for sub in lines])

        df_year=pd.concat([df_year, df], ignore_index=True)
        #Store the data file for a given year into a pickle file
    df_year.to_pickle("data/y"+_year+".pkl") 
    return df_year
'''
Fetches ATCF storm data for given years from the NOAA website
'''
def fetch_storm_data(_decks, INIT_DATA=False):
    if INIT_DATA:
        if os.path.exists("data/ATCF"):
            shutil.rmtree("data/ATCF")
    YEARS_TO_DOWNLOAD = [ 2023]  #2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
    YEARS_TO_DOWNLOAD = [str(x) for x in YEARS_TO_DOWNLOAD] #Convert integers to string '2010'......'2017'
    years = YEARS_TO_DOWNLOAD
    print(f"year to download{years}")
    #Get the list of URLs available on the FTP site by parsing the HTML web page
    for year in years:
        folder1 = 'data/ATCF/'+year+'/'
        #make a folder for the cyclone data if it does not exist
        # if the folder exists- do not download the data
        #C:\Users\xxx\Documents\Tropical Cyclones\Code\Project 8\NA\data\pressure
        if os.path.exists(folder1):
            create_year_picklefile(year)
            continue
        else:
            os.makedirs(folder1) 
        
        print(f"processing year {year}")
        # Scrapes a webpage to get list of all .tar.gz files. Each file contains all the satellite images associated
        # with a particular hurricane.
        year_directory_url = 'https://ftp.nhc.noaa.gov/atcf/archive/' + year
        year_directory_page = requests.get(year_directory_url).text
        year_directory_soup = BeautifulSoup(year_directory_page, 'html.parser')
        year_directory_file_urls=[]
        _file_gzs = [year_directory_url + '/' + node.get('href') for node in
                                    year_directory_soup.find_all('a') if node.get('href').endswith('dat.gz')]
        
        #Create a tuple of (year,[list of files for the year]) and append to the urls list
        year_directory_file_urls = year_directory_file_urls+[(year, _file_gzs)]
        for _deck in _decks:
            fetch_a_deck_for_a_year(_deck, year_directory_file_urls)
        
def fetch_a_deck_for_a_year(_deck, year_directory_file_urls):
    #Fetch the data for a - AL Basin
    for _yearurl in year_directory_file_urls:
        #Get the year part
        _year=_yearurl[0]
           
        #Get the URLs part
        _urls = _yearurl[1]
        
        for _url in _urls:
            _fname = _url.rsplit('/')[-1]
            
            #Get the files only starting with "aal"
            if(_fname[0:3]==_deck + 'al'):
                print(_fname)  
                storm_file_url = _url #'https://ftp.nhc.noaa.gov/atcf/archive/2018/aal012018.dat.gz'
                storm_file_path = 'data/ATCF/'+_year+'/'+_fname #'data/ATCF/aal012018.dat.gz'
                fetch_urls(storm_file_url, storm_file_path)
        
'''            
Fetch data corresponding to URLs and unpack it in local direcotry            
e.g. storm_file_url :  'https://ftp.nhc.noaa.gov/atcf/archive/2018/aal012018.dat.gz'
e.g. storm_file_path : 'data/ATCF/aal012018.dat.gz'
'''
def fetch_urls(storm_file_url, storm_file_path):
    
    file_name = storm_file_url.split('/')[-1]
    print(storm_file_path, file_name)
    #Get gz files to the local drive
    request = requests.get(storm_file_url, allow_redirects=True)  
    print("getting storm file:", storm_file_url)
    open(storm_file_path, 'wb').write(request.content)
    storm_dat_file = storm_file_path.rsplit('.', 1)[0]
    #unpack the gz file into .dat file
    with gzip.open(storm_file_path, 'rb') as f_in:
        with open(storm_dat_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    #Remove gz files since they are no longer needed
    os.remove(storm_file_path)
    
#Read *.dat file and store it in dataframe
def get_storm_data_from_datafile(YEARS, load_from_pickle = True):
    df_master = pd.DataFrame()
    
    #Get all the years in /data/ATCF directory
    for _year in YEARS:
        print(_year)
        if load_from_pickle:
            if os.path.exists("data/y"+_year+".pkl"):
                df_year = pd.read_pickle("data/y"+_year+".pkl") 
            else:
                print("File Not Found:", "data/y"+_year+".pkl") 
            
        else:
            
            df_year = create_year_picklefile(_year)
    
        #Append the year dataframe read to the master dataframe
        df_master=pd.concat([df_master, df_year], ignore_index=True)

    return df_master


#This function plots a sample forecast for a given date/time. 
def plot_a_forecast(df_ens, df_mean, df_best):
    _df = df_best.loc['2018-05-26'].between_time('12:00:00', '12:30:00')


    fig=px.scatter_geo(_df, lat='lat', lon='lon') #, color="tech")
    colors = ['black', 'green', 'brown', 'yellow', 'blue']

    for i in range(20):
        _df = df_ens.loc['2018-05-26'].between_time('12:00:00', '12:00:00')
        _df = _df.loc[_df['tech'] == 'AP'+ str(i).rjust(2, '0')]   
        fig.add_trace(go.Scattergeo(
                            lat=_df["lat"], 
                            lon=_df["lon"],
                            mode = 'lines',
                            line = dict(width = 2,color = colors[int(i/4)])
                        ))
    fig.update_geos(
        #title_text = 'Cyclone Paths',
        scope="usa")  #resolution=110, 
    fig.show()