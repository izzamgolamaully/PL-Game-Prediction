# Premier League Match Outcome Predictor - ENHANCED VERSION
# New features: ELO ratings, H2H records, advanced form, league positions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import glob
import os
from collections import defaultdict

def load_football_data(data_folder="./football_data/"):
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not all_files:
        raise ValueError(f"No CSV files found in {data_folder}")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            season = os.path.basename(file).replace('.csv', '')
            df['Season'] = season
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(all_files)} seasons with {len(combined_df)} total matches.")
    return combined_df

def preprocess_data(df):
    df = df.rename(columns={'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam', 
                            'FTHG': 'FTHG', 'FTAG': 'FTAG', 'FTR': 'FTR'})
    
    columns_needed = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season']
    df = df[[c for c in columns_needed if c in df.columns]]
    
    df['Result'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date', 'Result'])
    df.sort_values(['Season', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def calculate_elo_ratings(df, k_factor=20):
    """ELO ratings - the gold standard for team strength"""
    elo_ratings = defaultdict(lambda: 1500)
    
    df['HomeElo'] = 0.0
    df['AwayElo'] = 0.0
    df['EloDiff'] = 0.0
    
    for idx, row in df.iterrows():
        home_elo = elo_ratings[row['HomeTeam']]
        away_elo = elo_ratings[row['AwayTeam']]
        
        df.at[idx, 'HomeElo'] = home_elo
        df.at[idx, 'AwayElo'] = away_elo
        df.at[idx, 'EloDiff'] = home_elo - away_elo
        
        # Expected scores
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        
        # Actual result
        if row['Result'] == 1:
            actual_home = 1
        elif row['Result'] == 0:
            actual_home = 0.5
        else:
            actual_home = 0
        
        # Update ELO
        elo_ratings[row['HomeTeam']] += k_factor * (actual_home - expected_home)
        elo_ratings[row['AwayTeam']] += k_factor * ((1 - actual_home) - (1 - expected_home))
    
    return df

def calculate_head_to_head(df):
    """H2H win rates"""
    h2h_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    df['H2H_HomeWinRate'] = 0.0
    df['H2H_AwayWinRate'] = 0.0
    
    for idx, row in df.iterrows():
        matchup = (row['HomeTeam'], row['AwayTeam'])
        reverse = (row['AwayTeam'], row['HomeTeam'])
        
        if h2h_stats[matchup]['total'] > 0:
            df.at[idx, 'H2H_HomeWinRate'] = h2h_stats[matchup]['wins'] / h2h_stats[matchup]['total']
        
        if h2h_stats[reverse]['total'] > 0:
            df.at[idx, 'H2H_AwayWinRate'] = h2h_stats[reverse]['wins'] / h2h_stats[reverse]['total']
        
        # Update
        if row['Result'] == 1:
            h2h_stats[matchup]['wins'] += 1
        elif row['Result'] == -1:
            h2h_stats[reverse]['wins'] += 1
        
        h2h_stats[matchup]['total'] += 1
        h2h_stats[reverse]['total'] += 1
    
    return df

def calculate_form_features(df):
    """Form over last 3 and 5 matches, plus goals"""
    for col in ['HomeTeam_L3', 'AwayTeam_L3', 'HomeTeam_L5', 'AwayTeam_L5',
                'Home_GoalsL5', 'Away_GoalsL5', 'Home_ConcededL5', 'Away_ConcededL5']:
        df[col] = 0.0
    
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season].copy()
        
        df.loc[season_df.index, 'HomeTeam_L3'] = season_df.groupby('HomeTeam')['Result'].transform(
            lambda x: x.shift().rolling(3, min_periods=1).mean())
        df.loc[season_df.index, 'AwayTeam_L3'] = season_df.groupby('AwayTeam')['Result'].transform(
            lambda x: (-x).shift().rolling(3, min_periods=1).mean())
        df.loc[season_df.index, 'HomeTeam_L5'] = season_df.groupby('HomeTeam')['Result'].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean())
        df.loc[season_df.index, 'AwayTeam_L5'] = season_df.groupby('AwayTeam')['Result'].transform(
            lambda x: (-x).shift().rolling(5, min_periods=1).mean())
        
        # Goals
        df.loc[season_df.index, 'Home_GoalsL5'] = season_df.groupby('HomeTeam')['FTHG'].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean())
        df.loc[season_df.index, 'Away_GoalsL5'] = season_df.groupby('AwayTeam')['FTAG'].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean())
        df.loc[season_df.index, 'Home_ConcededL5'] = season_df.groupby('HomeTeam')['FTAG'].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean())
        df.loc[season_df.index, 'Away_ConcededL5'] = season_df.groupby('AwayTeam')['FTHG'].transform(
            lambda x: x.shift().rolling(5, min_periods=1).mean())
    
    return df

def create_all_features(df):
    print("Calculating ELO ratings...")
    df = calculate_elo_ratings(df)
    print("Calculating head-to-head records...")
    df = calculate_head_to_head(df)
    print("Calculating form features...")
    df = calculate_form_features(df)
    
    # Derived features
    df['HomeAdvantage'] = 1
    df['FormDiff'] = df['HomeTeam_L5'] - df['AwayTeam_L5']
    df['AttackDiff'] = df['Home_GoalsL5'] - df['Away_GoalsL5']
    df['DefenseDiff'] = df['Away_ConcededL5'] - df['Home_ConcededL5']
    
    return df

def train_model(df, test_season=None):
    features = ['HomeAdvantage', 'HomeElo', 'AwayElo', 'EloDiff',
                'H2H_HomeWinRate', 'H2H_AwayWinRate',
                'HomeTeam_L3', 'AwayTeam_L3', 'HomeTeam_L5', 'AwayTeam_L5',
                'Home_GoalsL5', 'Away_GoalsL5', 'Home_ConcededL5', 'Away_ConcededL5',
                'FormDiff', 'AttackDiff', 'DefenseDiff']
    
    df = df.dropna(subset=features)
    
    if test_season:
        train_df = df[df['Season'] != test_season]
        test_df = df[df['Season'] == test_season]
        X_train, y_train = train_df[features], train_df['Result']
        X_test, y_test = test_df[features], test_df['Result']
        print(f"\nTraining on {len(X_train)} matches, testing on {len(X_test)} matches")
    else:
        from sklearn.model_selection import train_test_split
        X, y = df[features], df['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5,
                                   random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
    
    if len(X_train) > 100:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    feat_imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print("\nTop 10 Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    top15 = feat_imp.head(15)
    plt.barh(range(len(top15)), top15['importance'])
    plt.yticks(range(len(top15)), top15['feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_enhanced.png', dpi=150)
    print("Plot saved: feature_importance_enhanced.png")
    
    return model, features

def simulate_season(df, model, features, season):
    season_df = df[df['Season'] == season].copy()
    if len(season_df) == 0:
        return None
    
    X = season_df[features]
    season_df['Predicted'] = model.predict(X)
    probs = model.predict_proba(X)
    season_df['Confidence'] = probs.max(axis=1)
    
    # Simulated standings
    team_pts = defaultdict(int)
    team_matches = defaultdict(int)
    for _, row in season_df.iterrows():
        team_matches[row['HomeTeam']] += 1
        team_matches[row['AwayTeam']] += 1
        if row['Predicted'] == 1:
            team_pts[row['HomeTeam']] += 3
        elif row['Predicted'] == 0:
            team_pts[row['HomeTeam']] += 1
            team_pts[row['AwayTeam']] += 1
        else:
            team_pts[row['AwayTeam']] += 3
    
    table = pd.DataFrame([{'Team': t, 'Points': p, 'Matches': team_matches[t]} 
                          for t, p in team_pts.items()])
    table = table.sort_values('Points', ascending=False).reset_index(drop=True)
    table.index += 1
    
    print(f"\nPredicted Standings ({season}):")
    print(table.to_string())
    
    # Actual standings
    actual_pts = defaultdict(int)
    for _, row in season_df.iterrows():
        if row['Result'] == 1:
            actual_pts[row['HomeTeam']] += 3
        elif row['Result'] == 0:
            actual_pts[row['HomeTeam']] += 1
            actual_pts[row['AwayTeam']] += 1
        else:
            actual_pts[row['AwayTeam']] += 3
    
    actual = pd.DataFrame([{'Team': t, 'Actual_Points': p} for t, p in actual_pts.items()])
    actual = actual.sort_values('Actual_Points', ascending=False).reset_index(drop=True)
    actual.index += 1
    
    print(f"\nActual Standings ({season}):")
    print(actual.to_string())
    
    # Stats
    correct = (season_df['Predicted'] == season_df['Result']).sum()
    high_conf = season_df[season_df['Confidence'] > 0.6]
    hc_correct = (high_conf['Predicted'] == high_conf['Result']).sum() if len(high_conf) > 0 else 0
    
    print(f"\nMatch Accuracy: {correct/len(season_df):.1%} ({correct}/{len(season_df)})")
    print(f"High Confidence (>60%): {hc_correct/len(high_conf):.1%} ({hc_correct}/{len(high_conf)})" if len(high_conf) > 0 else "")
    print(f"Avg Confidence: {season_df['Confidence'].mean():.1%}")

if __name__ == "__main__":
    df = load_football_data("./football_data/")
    df = preprocess_data(df)
    df = create_all_features(df)
    
    seasons = sorted(df['Season'].unique())
    print(f"\nSeasons: {', '.join(seasons)}")
    
    test_season = seasons[-1]
    print(f"\n{'='*60}\nTRAINING ENHANCED MODEL\n{'='*60}")
    model, features = train_model(df, test_season=test_season)
    
    print(f"\n{'='*60}\nSEASON SIMULATION\n{'='*60}")
    simulate_season(df, model, features, test_season)