# Premier League Match Outcome Predictor & Season Simulator using Random Forest
#import libraries
from xml.parsers.expat import model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
import glob,os

from sympy import rf



#loading multiple seasons of premier league csv files
def load_football_data(data_folder = "./football_data/"):
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        raise ValueError("No CSV files found in {data_folder}. Please download season data from football-data.co.uk")
   
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df['SeasonFile'] = file.split('/')[-1]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_files)} seasons with {len(combined_df)} total matches.")
    return combined_df

#Cleaning and feature engineering
def preprocess_data(df):
    #standardizing column names
    rename_map = {
        'HomeTeam': 'HomeTeam',
        'AwayTeam': 'AwayTeam', 
        'FTHG': 'FTHG',
        'FTAG': 'FTAG',
        'FTR': 'FTR',
        'HS': 'HS',
        'AS': 'AS',
        'HST': 'HST',
        'AST': 'AST',
        'HC': 'HC',
        'AC': 'AC',
    }

    df = df.rename(columns=rename_map)

    #keep only key columns
    columns_needed = ['Date' ,'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
    df = df[[ c for c in columns_needed if c in df.columns]]

    #encoding match results 1=Home Win, 0=Draw, -1=Away Win
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

    #handling missing values or malformed dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.sort_values('Date', inplace=True)

    #Home advantage
    df['HomeAdvantage'] = 1

    #rolling form based on last 5 matches
    df['HomeTeamForm'] = df.groupby('HomeTeam')['Result'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['AwayTeamForm'] = df.groupby('AwayTeam')['Result'].apply(lambda x: (-x).shift().rolling(5, min_periods=1).mean())

    df.dropna(inplace=True)
    return df

#model training
def train_random_forest(df):
    features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HomeAdvantage', 'HomeTeamForm', 'AwayTeamForm']
    df= df.dropna(subset=features)
    X = df[features]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nClassification Report: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test, y_pred))
    #print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return model