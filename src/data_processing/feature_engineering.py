import pandas as pd
from tqdm import tqdm
import numpy as np
tqdm.pandas()
import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')
import warnings
warnings.filterwarnings("ignore")


paths_pro = {
    'matches_csv': 'src/data/processed/all_match_data_updated.csv',
    'player_stats_csv': 'src/data/processed/all_player_stats.csv',
    'json_dir': 'src/data/raw/cricksheet_data',
    'roles': 'src/data/processed/new_roles_df.csv',
    'features':'src/data/processed/full_data_features.csv'
}


def generate_time_series_features(group, lags=3, rolling_windows=[3, 7], timestamp_col=None):
    features = pd.DataFrame({
        'match_id': group['match_id'],
        'player_id': group['player_id'],
        'team':group['team'],
        'name':group['player_name'],
        'match_date': group['date'],
        'format': group['format'],
        'target': group['fantasy_points'],
        'f_pos':group['f_pos']
    })
    columns = [
        'runs_scored','fours', 'sixes', 'strike_rate', 'wickets', 'maiden_overs', 'economy', 'catches','stumpings'
    ]
    columns2 = ['fantasy_points']

    for column in columns2+columns:
        for lag in range(1, lags + 1):
            features[f'{column}_lag_{lag}'] = group[column].shift(lag)
    for column in columns2:
        for window in rolling_windows:
            features[f'{column}_rolling_mean_{window}'] = group[column].rolling(window=window).mean()
            features[f'{column}_rolling_std_{window}'] = group[column].rolling(window=window).std()
            features[f'{column}_rolling_min_{window}'] = group[column].rolling(window=window).min()
            features[f'{column}_rolling_max_{window}'] = group[column].rolling(window=window).max()
            features[f'{column}_rolling_var_{window}'] = group[column].rolling(window=window).var()
            features[f'{column}_rolling_sum_{window}'] = group[column].rolling(window=window).sum()
    for column in columns2+columns:
        # EWMA features
        for span in rolling_windows:
            features[f'{column}_ewma_{span}'] = group[column].ewm(span=span, adjust=False).mean()

        # Difference features
        for diff in range(1, lags + 1):
            features[f'{column}_diff_{diff}'] = group[column].diff(diff)

        # Cumulative statistics
        features[f'{column}_cumulative_mean'] = group[column].expanding().mean()
        features[f'{column}_cumulative_std'] = group[column].expanding().std()
        features[f'{column}_cumulative_var'] = group[column].expanding().var()

    def get_opposition(row):
        return (row['info_teams_1'] if row['team'] == row['info_teams_0'] 
               else row['info_teams_0'])
    
    group['opposition'] = group.apply(get_opposition, axis=1)
    for column in ['fantasy_points','runs_scored','wickets']:
        features[f'format_avg_{column}'] = (
            group.groupby('format',group_keys=False)[column]
            .expanding()  # Calculate the expanding mean within each group
            .mean()
            .reset_index(level=0, drop=True)
        )
        features[f'venue_avg_{column}'] = (
            group.groupby('info_venue',group_keys=False)[column]  # Shift to exclude the current row from the mean
            .expanding()  # Calculate the expanding mean within each group
            .mean()
            .reset_index(level=0, drop=True)
        )
        features[f'opp_avg_{column}'] = (
            group.groupby('opposition',group_keys=False)[column]  # Shift to exclude the current row from the mean
            .expanding()  # Calculate the expanding mean within each group
            .mean()
            .reset_index(level=0, drop=True)
        )
    features = features.reset_index(drop=True)
    shift_cols =[col for col in features.columns if col not in ['match_id','player_id','match_date','target','format','team','name','f_pos']]
    features.loc[:,shift_cols] = features[shift_cols].shift(1)
    features = features.fillna(0)
    
    return features


def f_create_player_features(player_df,match_df,feature_df):
    player_df = player_df.copy()
    match_df = match_df.copy()
    feature_df = feature_df.copy()
    enriched_stats = player_df.merge(
        match_df[['match_id', 'info_venue', 'info_teams_0', 'info_teams_1', 'info_event_name']],
        on='match_id'
    )
    tqdm.pandas()
    latest_date = pd.to_datetime(feature_df["match_date"].max())
    new_p = list(set(enriched_stats[enriched_stats['date']>=latest_date]['player_id']))
    enriched_stats = enriched_stats[enriched_stats['player_id'].isin(new_p)].reset_index(drop=True)
    old_feature_df= feature_df[~(feature_df['player_id'].isin(new_p))].reset_index(drop=True)
    enriched_stats = enriched_stats.sort_values(['player_id', 'date'])
    features=[]
    grouped = enriched_stats.groupby('player_id')
    for i,g in tqdm(grouped):
        features.append(generate_time_series_features(g))
    features= pd.concat(features)
    features = features.reset_index(drop=True)
    # features = features.dropna(subset='player_id')
    # features = features.merge(player_df[['player_id','is_batter','is_bowler','is_wicketkeeper','is_allrounder']],on='player_id')
    features = features.merge(match_df[['match_id','info_gender']],on='match_id',how='left')
    features = features.reset_index(drop=True)
    new_features_df = pd.concat([old_feature_df,features],ignore_index=True)
    new_features_df.to_csv(paths_pro['features'],index=False)
    return new_features_df


def get_features():
    player_df = pd.read_csv(paths_pro['player_stats_csv'],low_memory=False)
    match_df = pd.read_csv(paths_pro['matches_csv'],low_memory=False)
    full_data = pd.read_csv(paths_pro['features'],low_memory=False)
    # assert 0

    player_df['date'] = pd.to_datetime(player_df['date'])
    match_df['date'] = pd.to_datetime(match_df['info_dates_0'])
    full_data['match_date'] = pd.to_datetime(full_data['match_date'])

    player_df = player_df.sort_values(by='date')
    print(set(player_df['match_id'])-set(match_df['match_id']))



    full_data = f_create_player_features(player_df,match_df,full_data)
    return full_data

# get_features()