import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

'''

New File formats are available at https://www.nrlmry.navy.mil/atcf_web/docs/database/new/database.html
Video explains https://www.nrlmry.navy.mil/atcf_web/docs/wmv/Web-ATCF-Intro.mp4

https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
BASIN, CY, YYYYMMDDHH, TECHNUM/MIN, TECH, TAU, LatN/S, LonE/W, VMAX, MSLP, TY, RAD, WINDCODE, RAD1, RAD2, RAD3, RAD4, POUTER, ROUTER, RMW, GUSTS, EYE, SUBREGION, MAXSEAS, INITIALS, DIR, SPEED, STORMNAME, DEPTH, SEAS, SEASCODE, SEAS1, SEAS2, SEAS3, SEAS4, USERDEFINED, userdata

COMMON FIELDS

BASIN      - basin, e.g. WP, IO, SH, CP, EP, AL, LS
CY         - annual cyclone number: 1 - 99
YYYYMMDDHH - Warning Date-Time-Group, yyyymmddhh: 0000010100 through 9999123123.
TECHNUM/MIN- objective technique sorting number, minutes for best track: 00 - 99
TECH       - acronym for each objective technique or CARQ or WRNG,
             BEST for best track, up to 4 chars.
TAU        - forecast period: -24 through 240 hours, 0 for best-track, 
             negative taus used for CARQ and WRNG records.
LatN/S     - Latitude for the DTG: 0 - 900 tenths of degrees,
             N/S is the hemispheric index.
LonE/W     - Longitude for the DTG: 0 - 1800 tenths of degrees,
             E/W is the hemispheric index.
VMAX       - Maximum sustained wind speed in knots: 0 - 300 kts.
MSLP       - Minimum sea level pressure, 850 - 1050 mb.
TY         - Highest level of tc development:
             DB - disturbance, 
             TD - tropical depression, 
             TS - tropical storm, 
             TY - typhoon, 
             ST - super typhoon, 
             TC - tropical cyclone, 
             HU - hurricane, 
             SD - subtropical depression,
             SS - subtropical storm,
             EX - extratropical systems,
             PT - post tropical,
             IN - inland,
             DS - dissipating,
             LO - low,
             WV - tropical wave,
             ET - extrapolated,
             MD - monsoon depression,
             XX - unknown.
RAD        - Wind intensity for the radii defined in this record: 34, 50 or 64 kt.
WINDCODE   - Radius code:
             AAA - full circle
             NEQ, SEQ, SWQ, NWQ - quadrant 
RAD1       - If full circle, radius of specified wind intensity, or radius of
             first quadrant wind intensity as specified by WINDCODE.  0 - 999 n mi
RAD2       - If full circle this field not used, or radius of 2nd quadrant wind
             intensity as specified by WINDCODE.  0 - 999 n mi.
RAD3       - If full circle this field not used, or radius of 3rd quadrant wind
             intensity as specified by WINDCODE.  0 - 999 n mi.
RAD4       - If full circle this field not used, or radius of 4th quadrant wind
             intensity as specified by WINDCODE.  0 - 999 n mi.
POUTER     - pressure in millibars of the last closed isobar, 900 - 1050 mb.
ROUTER     - radius of the last closed isobar, 0 - 999 n mi.
RMW        - radius of max winds, 0 - 999 n mi.
GUSTS      - gusts, 0 - 999 kt.
EYE        - eye diameter, 0 - 120 n mi.
SUBREGION  - subregion code: W,A,B,S,P,C,E,L,Q.
             A - Arabian Sea
             B - Bay of Bengal
             C - Central Pacific
             E - Eastern Pacific
             L - Atlantic
             P - South Pacific (135E - 120W)
             Q - South Atlantic
             S - South IO (20E - 135E)
             W - Western Pacific
MAXSEAS    - max seas: 0 - 999 ft.
INITIALS   - Forecaster's initials used for tau 0 WRNG or OFCL, up to 3 chars.
DIR        - storm direction, 0 - 359 degrees.
SPEED      - storm speed, 0 - 999 kts.
STORMNAME  - literal storm name, number, NONAME or INVEST, or TCcyx where:
             cy = Annual cyclone number 01 - 99
             x  = Subregion code: W,A,B,S,P,C,E,L,Q.
DEPTH      - system depth, 
	     D - deep, 
	     M - medium, 
	     S - shallow, 
	     X - unknown
SEAS       - Wave height for radii defined in SEAS1 - SEAS4, 0 - 99 ft.
SEASCODE   - Radius code:
             AAA - full circle
             NEQ, SEQ, SWQ, NWQ - quadrant 
SEAS1      - first quadrant seas radius as defined by SEASCODE,  0 - 999 n mi.
SEAS2      - second quadrant seas radius as defined by SEASCODE, 0 - 999 n mi.
SEAS3      - third quadrant seas radius as defined by SEASCODE,  0 - 999 n mi.
SEAS4      - fourth quadrant seas radius as defined by SEASCODE, 0 - 999 n mi.
USERDEFINE1- 1 to 20 character description of user data to follow.
userdata1  - user data section as indicated by USERDEFINED parameter (up to 100 char).
USERDEFINE2- 1 to 20 character description of user data to follow.
userdata2  - user data section as indicated by USERDEFINED parameter (up to 100 char).
USERDEFINE3- 1 to 20 character description of user data to follow.
userdata3  - user data section as indicated by USERDEFINED parameter (up to 100 char).
USERDEFINE4- 1 to 20 character description of user data to follow.
userdata4  - user data section as indicated by USERDEFINED parameter (up to 100 char).
USERDEFINE5- 1 to 20 character description of user data to follow.
userdata5  - user data section as indicated by USERDEFINED parameter (up to 100 char).

'''
def create_best_dataframe(year, cy, df_best1, x):
    a = df_best1.loc[df_best1.index.year == year]
    df_best1 = a[a['CY'] == cy]
    _df1 = df_best1[['lat', 'lon']]
    _df2 = _df1.shift(-1)
    df = pd.merge(_df1, _df2, left_index=True, right_index=True)

    for i in range (12, 120, 6):
        _df2 = _df2.shift(-1)
        df = pd.merge(df, _df2, left_index=True, right_index=True)

    x = pd.concat([x, df])
    return x

def create_best_matrix(df_best1):
    x = pd.DataFrame()
    for year in df_best1.index.year.unique():
        for cy in df_best1['CY'].unique():
                x = create_best_dataframe(year, cy, df_best1, x)
                
    return x

def create_matrix1(df_ens, df_best):
    _df_ens_master = pd.DataFrame()
    for i in df_ens.index.year.unique():
        #Get an ensemble forecast and best track in a given year
        df_ens1 = df_ens.loc[df_ens.index.year==i]  
        df_best1 = df_best.loc[df_best.index.year==i]  
        
        for j in df_ens1['CY'].unique():
            #Get an ensemble forecast and best track for a cyclone
            df_ens2 = df_ens1.loc[df_ens1['CY']==j] 
            df_best2 = df_best1.loc[df_best1['CY_x']==j] 
            
            #Get only the lat and lon columns from best track - these will be the last 2 columns
            #df_best2 = df_best2[['lat', 'lon']]
            
            #Drop duplicated and na values.
            df_best2 = df_best2.drop_duplicates()
            df_best2 = df_best2.dropna()
            _df = create_matrix_row(df_ens2)
            
            #The last 2 columns are that of Best tracks
            #CY_, tech_, lat_0.0, lon_0.0, lat_6.0, lon_6.0, lat_12.0, lon_12.0, lat, lon
            _df=pd.merge(_df, df_best2, left_index=True, right_index=True) #.loc[:, ['lat', 'lon']]
                    
            _df_ens_master = pd.concat([_df_ens_master, _df])
    
    # We don't need the date and CY columns --- Check with the CNN model X input
    _df_ens_master=_df_ens_master.reset_index()
    _df_ens_master=_df_ens_master.drop(['date', 'CY'], axis=1)
    return _df_ens_master
        
def create_matrix_row(df_ens2):
    df_ens3=df_ens2[['CY','tech','tau','lat','lon']]
    df_ens3=df_ens3[df_ens3['tau'].isin([0.0, 6.0, 12.0])]
    df_ens3['tau']=df_ens3['tau'].astype(str)
    df_ens3.reset_index(inplace=True)
    df_ens3.set_index(['date','CY','tech'], inplace=True)
    df_ens3 = df_ens3.sort_values(['date','CY','tech'], ascending=True)
    
    df_ens3=df_ens3.drop_duplicates( keep='first', ignore_index=False)
    df_ens3 = df_ens3.reset_index()
    
    #Flatten out tau's 0.0, 6.0, 12.0...Remember tau column names are converted into string format
    #null values are filled with bfill otherwise the CNN model breaks for null values
    df_ens3=df_ens3.pivot(index=['date','CY'],columns=['tau','tech'])
    
    
    #Flatten the multindex out to single level. Column level names are appended with lat/lon_tau_tech
    df_ens3.columns = ['_'.join(col).strip() for col in (df_ens3.columns)]
    _df = df_ens3.reset_index() #inaplce =True does not work here
    
    #Set the null values to the mean of the column, need to set this as mean of ROW????
    _df=_df.fillna(_df.mean(), axis=0)
    
    #Add 18 Hours since this new date/time is the timestamp of best track predcited based on prior 18 hours ensemble forecasts
    _df['date'] = _df['date'] + timedelta(hours = 18.0)
    
    _df.set_index('date', inplace=True)
        
    return _df


def generate_ensemble_forecasts(df_master):
    df_New = df_master.drop(df_master.iloc[:, 11:],
                       axis = 1)
    #_col = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    #df_New = df_master[_col]
    df_New.columns =['basin', 'CY', 'date', 'techNum', 'tech', 'tau', 'lat', 'lon', 'vmax', 'MSLP', 'TY']
    
    #The below line gives memory error
    #df_New[df_New.columns] = df_New.apply(lambda x: x.str.strip())  #The columns have empty spaces - so get rid of it
    df_New['basin']= df_New['basin'].str.replace(' ', '')
    df_New['CY']= df_New['CY'].str.replace(' ', '')
    df_New['date'] = df_New['date'].str.replace(' ', '')
    df_New['techNum'] = df_New['techNum'].str.replace(' ', '')
    df_New['tech'] = df_New['tech'].str.replace(' ', '')
    df_New['tau'] = df_New['tau'].str.replace(' ', '')
    df_New['lat'] = df_New['lat'].str.replace(' ', '')
    df_New['lon'] = df_New['lon'].str.replace(' ', '')
    df_New['vmax'] = df_New['vmax'].str.replace(' ', '')
    df_New['MSLP'] = df_New['MSLP'].str.replace(' ', '')
    df_New['TY'] = df_New['TY'].str.replace(' ', '')


    df_New['tau']= df_New['tau'].astype(float)
    #df_New['c'].str.strip()
    df_New.set_index('date', inplace=True)
    #df_New['c']=pd.to_datetime(df_New.c, format="%Y%m%d%H")
    df_New.index = pd.to_datetime(df_New.index, format="%Y%m%d%H")
    #The following line checks if there is N at the end, if so takes out the positive value
    # and multiplies it with 1 otherwise multiplies it with -1
    df_New['lat'] =np.where(df_New['lat'].str[-1]=='N', 
                             df_New['lat'].str[:-1].astype(float).fillna(0.0)/10, 
                             df_New['lat'].str[:-1].astype(float).fillna(0.0)/10*-1)

    #The following line checks if there is N at the end, if so takes out the positive value
    # and multiplies it with 1 otherwise multiplies it with -1
    df_New['lon'] =np.where(df_New['lon'].str[-1]=='W', 
                             df_New['lon'].str[:-1].astype(float).fillna(0.0)/10*-1, 
                             df_New['lon'].str[:-1].astype(float).fillna(0.0)/10)

    df_New = df_New.loc[df_New['tau'] <= 120]
    _ens = ['BEST','AC00', 'AEMN', 'AP01', 'AP02', 'AP03',
           'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11',
           'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19',
           'AP20']
    df_New=df_New.loc[df_New['tech'].isin(_ens)]

    print(df_New)
    
    #We create 3 dataframes - ensemble forecasts, mean, Best Forecasts
    df_best = df_New.loc[df_New['tech'] == 'BEST'].copy()
    df_mean = df_New.loc[df_New['tech'] == 'AEMN'].copy()
    #Delete BEST forecast from orignal dataframe
    df_ens = df_New[(df_New.tech != 'BEST') & (df_New.tech != 'AEMN')]
    #Only keep dates/times in the original dataframe (ensemble) and mean for which Best Track is available
    df_ens = df_ens[df_ens.index.isin(df_best.index)]
    df_ens_sorted = df_ens.sort_index()
    df_mean = df_mean[df_mean.index.isin(df_best.index)]
    return df_ens_sorted, df_mean, df_best