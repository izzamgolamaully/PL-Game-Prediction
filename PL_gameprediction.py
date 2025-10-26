# Premier League Match Outcome Predictor & Season Simulator using Random Forest
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import soccerdata as sd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#loading football-data.co.uk csv files
FOOTBALL_DATA_PATH = './football_data/' # folder with downloaded E0_*.csv filesimport glob,os
import glob,os

def load_football_data(path):
    files = glob.glob(os.path.join(path, 'E0_*.csv'))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.rename(columns = {'Date':'date','HomeTeam':'home_team','AwayTeam':'away_team','FTHG':'home_goals', 'FTAG':'away_goals', 'FTR':'result'})

        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        dfs.append(df)
    
    full = pd.concat(dfs, ignore_index=True)
    full = full.sort_values('date').reset_index(drop=True)
    return full

fd = load_football_data(FOOTBALL_DATA_PATH)
fd = fd[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']]
print('Football data loaded with shape:', fd.shape)

#Fetching fbref stats
LEAGUE = 'ENG-PREMIER-LEAGUE'
fb = sd.FBref(LEAGUE,None)
fb_schedule = fb.read_schedule()
fb_stats = fb.read_match_stats()
fb_df = fb_schedule.merge(fb_stats, on=['home_team','away_team','date'], how='left')
print('FBref shape:', fb_df.shape)

#Merging datasets
df = pd.merge(fd, fb_df, on=['date','home_team','away_team'], how='left')
df = df[~df['home_goals'].isna()].reset_index(drop=True)
print('Merged dataset shape:', df.shape)

#Pre-match Elo rating
