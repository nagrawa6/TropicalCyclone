import pandas as pd
import numpy as np

def process_coordinates(_coordinate_type, _df, NUM_OF_INTERVALS):
    _df.reset_index(inplace=True)
    _df.set_index(['date', 'basin', 'CY'], inplace=True)
    #_coordinate_type could be 'lat', 'lon'
    _df=_df[['tech', 'tau', _coordinate_type]]  
    table = pd.pivot_table(_df, index=['date', 'basin', 'CY'],
                           columns=['tech','tau'], aggfunc=np.mean)
    
    #Generate column names for lat and then lon 
    #AC00lat00 AC00lat06 AC00lat12 ....
    _c = table.columns
    table.columns = [b+a+str(int(c)).zfill(2) for (a, b, c) in _c]

    #This a routine used from stacked overflow. This replaces nan with mean values for row.
    #each is replaced with mean value for lat or lon depending on type of coordinaete type passed
    mask = np.isnan(table)
    masked_df = np.ma.masked_array(table, mask)
    fill_value = pd.DataFrame({col: table.mean(axis=1) for col in table.columns})
    table_wmean = masked_df.filled(fill_value)
    _df = pd.DataFrame(data=table_wmean)

    #Important: We get ['date','tau'] back since the above routine drops these colums
    #so we need to put them back here. Given dimensions matching requirement -> it is done here.
    _df.index=table.index
    _df.columns = table.columns
    _df = _df.astype(float).round(1)
    
    return _df


def generate_ensemble_track(_de1, NUM_OF_INTERVALS):
    #We consider only taus within )RANGE defined globally
    _range = [float(x) for x in range(0, NUM_OF_INTERVALS*6, 6)]

    _de1 = _de1.loc[_de1['tau'].isin(_range)]
    _de1.reset_index(inplace=True)
    #Set the index to date, basin, CY and tau which is used to pivot table based upon later
    _de1.set_index(['date', 'basin', 'CY', 'tech'], inplace=True)
    _de1 = _de1[['tau', 'lat', 'lon']]
    _de1 = _de1.sort_values(by=['CY','date','tau','tech'])
    #We process it for lat and lon separately since we need replace nan with mean(lat or lon)
    _df_lat = process_coordinates('lat', _de1, NUM_OF_INTERVALS)
    _df_lon = process_coordinates('lon', _de1, NUM_OF_INTERVALS)
    table = pd.concat([_df_lat,_df_lon], keys=['date', 'basin', 'CY'], axis=1)

    #We do this remove the additional level that gets added in 'date'
    table.columns  = table.columns.droplevel()
    
    #This is an important step. By shifting 1 level forward, we using prior ensemble's (6 hours before) prediction
    #for t=0 onward track. 
    table = table.shift(1)
    #We delete the first row since it is redundant.
    table = table.iloc[1:]    
    return table

#Generate mean track for tau upto the values in range
def generate_mean_track(_dfm, NUM_OF_INTERVALS):
    _range = [float(x) for x in range(0, NUM_OF_INTERVALS*6, 6)]
    _dm1 = _dfm.loc[_dfm['tau'].isin(_range)]
    _dm1.reset_index(inplace=True)
    _dm1.set_index(['date', 'basin', 'CY', 'tech'], inplace=True)
    _dm1 = _dm1[['tau', 'lat', 'lon']]
    #Here we pivot on Tech and Tau so they move to columns
    _dm = pd.pivot_table(_dm1, values=['lat', 'lon'], index=['date', 'basin', 'CY'],
                               columns=['tech','tau'], aggfunc=np.mean)
    _c = _dm.columns
    #We construct column names as AC00lat00, AC00lat06, ....
    #Tuple columns names were stored as ('lat', 'tech' 'tau')
    _dm.columns = [b+a+str(int(c)).zfill(2) for (a, b, c) in _c]
    _dm = _dm.dropna(axis=0, thresh=NUM_OF_INTERVALS) 
    #This is an important step. By shifting 1 level forward, we using prior ensemble's (6 hours before) prediction
    #for t=0 onward track. 
    _dm = _dm.shift(1)
    #We delete the first row since it is redundant.
    _dm = _dm.iloc[1:]    

    return _dm

#Generate best track for tau upto the values in range
def generate_best_track(_dfb, NUM_OF_INTERVALS):
    df_best1 = _dfb.drop_duplicates() #(subset=['date'])
    #certain data items in df_best are duplicated in other columns which was throwing dimension mismatch error.
    df_best1 = df_best1[~df_best1.index.duplicated()]

    _min = df_best1.index.min()
    #We use this to create a date range for the dataframe.
    #This is done if Best track is missing for intermediate dates, those dates will still be captured
    #Necessary to create exact 6,12,18 hours before data points
    _max = df_best1.index.max()
    _index = pd.date_range (start=_min, end=_max, freq='6H')
    _nrows = len(_index)
    # Creates pandas DataFrame.
    _df_merged = pd.DataFrame(index=_index)
    _df_merged.index.name = 'date'
    #print(f"Before merge _df_merged size {_df_merged.shape}, best_track size:{df_best1.shape}")
    _df_merged = pd.merge(_df_merged, df_best1, how='left', on='date') #left_index=True, right_index=True)
    
    
    
    _df = _df_merged[['lat', 'lon', 'CY']]
    _df.index.name='date'
    
    _range = [float(x) for x in range(0, NUM_OF_INTERVALS*6, 6)]

    _db1 = _dfb.loc[_dfb['tau'].isin(_range)]
    _db1.reset_index(inplace=True)
    _db1.set_index(['date', 'basin', 'CY'], inplace=True)
    _db1 = _db1[['lat', 'lon']]
    _db1.columns = ['BESTLAT00', 'BESTLON00']
    _db = [_db1]
    for i in range(1,NUM_OF_INTERVALS):
        _db1 = _db1.shift(-1)
        _db1.columns = ['BESTLAT'+ str(i*6).zfill(2),
                        'BESTLON'+ str(i*6).zfill(2)]
        _db.append(_db1)
    _db=pd.concat(_db, axis=1)
    _db=_db.dropna(axis=0, thresh=NUM_OF_INTERVALS) #Atleast half the columns are not nulls
    
    #Had to remove duplicated index again for some reason
    _db = _db.drop_duplicates() #(subset=['date'])
    #certain data items in df_best are duplicated in other columns which was throwing dimension mismatch error.
    _db1 = _db[~_db.index.duplicated()]

    return _df_merged, _db1   
    
    
#Generates X train data: 
# First N columns are 21x3x2 = 126 columns for 21 forecasts, 3 prior data points, lat/lon coordinates
def generate_X_train_data(df_ens, df_mean, df_best, _NUM_OF_INTERVALS):
    NUM_OF_INTERVALS = _NUM_OF_INTERVALS
    df_best1=df_best.loc[(df_best['basin']=='AL')].drop_duplicates()  # & (df_best1['CY']=='01')
    df_ens1=df_ens.loc[(df_ens['basin']=='AL')] #&  & (df_ens1['CY']=='01')
    df_mean1=df_mean.loc[(df_mean['basin']=='AL')].drop_duplicates() 
    df_master_X = pd.DataFrame()
    df_bmaster_X = pd.DataFrame()
    df_mmaster_X = pd.DataFrame()
    for i in df_ens1.index.year.unique():
        #Get an ensemble forecast and best track in a given year
        df_ens2 = df_ens1.loc[df_ens1.index.year==i]  
        df_best2 = df_best1.loc[df_best1.index.year==i]  
        df_mean2 = df_mean1.loc[df_mean1.index.year==i]  
        df_best2=df_best2.sort_values(by=['date','tau'])
        df_ens2=df_ens2.sort_values(by=['date','tau'])
        df_mean2=df_mean2.sort_values(by=['date','tau'])
        #Only use in the increments of 6 hours forecast
        df_best2 =df_best2.loc[df_best2.index.hour%6==0]
        df_ens2 =df_ens2.loc[df_ens2.index.hour%6==0]
        df_mean2 =df_mean2.loc[df_mean2.index.hour%6==0]
        
        for j in df_ens2['CY'].unique():
            #print(j)
            #Get an ensemble forecast and best track for a cyclone
            df_ens3 = df_ens2.loc[df_ens2['CY']==j] 
            df_best3 = df_best2.loc[df_best2['CY']==j] 
            df_best3 = df_best3.drop_duplicates()
            df_mean3 = df_mean2.loc[df_mean2['CY']==j] 
            df_mean3 = df_mean3.drop_duplicates()
            
            #We concat ensemble and best track. 
            _df_merged, _db =generate_best_track(df_best3, NUM_OF_INTERVALS)
            #If there is no best or mean track, we do not want to consider the data
            if(_db.shape[0]>0):
                pass
            else:
                continue
            #Generate ensemble track with date row to a prior ensemble forecast
            _de = generate_ensemble_track(df_ens3, NUM_OF_INTERVALS)
            #If there is no best or mean track, we do not want to consider the data
            if(_de.shape[0]>0):
                #Generate best track
                #print(f"{i}{j}de shape:{_de.shape[0]}, db shape:{_db.shape[0]}")
                _x = _de
            else:
                continue
            #We concated mean only if there are values in it
            if(df_mean3.shape[0]>0):
                _dm =generate_mean_track(df_mean3, NUM_OF_INTERVALS)
                _x=pd.concat([_x,_dm], axis=1)
            #We concatenate best track at the end
            _x=pd.concat([_x,_db], axis=1)
           
            '''
            new_columns = [col + "-18" for col in _x.shift(2).iloc[:, :-42].columns]
            new_columns = new_columns + [col + "-12" for col in _x.shift(1).iloc[:, :-42].columns]
            new_columns = new_columns + [col + "-06" for col in _x.iloc[:, :-42].columns]
            new_columns = new_columns + [col for col in _x.iloc[:, -42:].columns] 

            _x = pd.concat([_x.shift(2).iloc[:, :-42], _x.shift(1).iloc[:, :-42], _x], axis = 1)
            _x = pd.concat([_x.shift(1).iloc[:, :-42], _x], axis = 1)
            _x.columns = new_columns
            '''
                #Add to the list vertically    
            df_master_X = pd.concat([df_master_X, _x], axis=0)
    
    ###IMportant: Here we are eliminating rows with BEST tracks set to NaN
    ##First we get the column names that start with BEST
    ##Next we set Thresh = 2*NUMBER of INtervals. 2 for lat and lon
    _col = df_master_X.columns.str.startswith ('BEST')
    df_master_X = df_master_X.dropna(subset = df_master_X.columns[_col], axis=0, thresh=NUM_OF_INTERVALS*2)

    
    
    return df_master_X