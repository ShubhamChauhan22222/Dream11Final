data_dir = 'https://cricsheet.org/downloads/all_json.zip'

import os
import shutil
import pandas as pd
import requests
import zipfile
import pandas as pd
import json
from tqdm import tqdm
import sys
from collections import defaultdict
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
import re
import numpy as np
import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')
print('hello')

paths = {
    'matches_csv': 'src/data/interim/all_match_data_updated.csv',
    'player_stats_csv': 'src/data/interim/all_player_stats.csv',
    'json_dir': 'src/data/raw/cricksheet_data/json_files'
}
paths_pro = {
    'matches_csv': 'src/data/processed/all_match_data_updated.csv',
    'player_stats_csv': 'src/data/processed/all_player_stats.csv',
    'json_dir': 'src/data/raw/cricksheet_data/json_files',
    'roles': 'src/data/processed/new_roles_df.csv'
}
required_roles = {'Batter', 'Bowler', 'Allrounder', 'Wicketkeeper'}


def setup_kaggle_data(dataset_path="src/data/interim"):
    """
    Copy required files from Kaggle dataset to working directory and setup paths
    """
    # Setup working directory paths
    INTER_DIR = "src/data/interim"
    CRICKSHEET_DIR= 'src/data/raw/cricksheet_data'
    MATCHES_CSV = os.path.join(INTER_DIR, "all_match_data_updated.csv")
    PLAYER_STATS_CSV = os.path.join(INTER_DIR, "all_player_stats.csv")
    JSON_DIR = os.path.join(CRICKSHEET_DIR, "json_files")
    
    # Create json directory if it doesn't exist
    os.makedirs(JSON_DIR, exist_ok=True)
    
    # Copy matches info CSV
    src_matches_csv = os.path.join(dataset_path, "all_match_data_updated.csv")
    if os.path.exists(src_matches_csv):
        print(f"{MATCHES_CSV} exists")
    else:
        print("Creating new matches CSV file")
        pd.DataFrame().to_csv(MATCHES_CSV, index=False)
    
    # Copy player stats CSV if exists
    src_player_stats = os.path.join(dataset_path, "all_player_stats.csv")
    if os.path.exists(src_player_stats):
        print(f"{PLAYER_STATS_CSV} exists")
    else:
        print("Creating new player stats CSV file")
        pd.DataFrame().to_csv(PLAYER_STATS_CSV, index=False)
    
    # Copy JSON files
    
    url = data_dir
    zip_path = data_dir.split('/')[-1]
    response = requests.get(url)
    with open(zip_path, "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(JSON_DIR)
    os.remove(zip_path)
    
    return {
        'matches_csv': MATCHES_CSV,
        'player_stats_csv': PLAYER_STATS_CSV,
        'json_dir': JSON_DIR
    }

def load_data():
    # Adjust the dataset path according to your Kaggle dataset
    global paths
    paths = setup_kaggle_data("src/data/interim")
    print("\nPaths setup:")
    for key, path in paths.items():
        print(f"{key}: {path}")




def flatten_json(data, prefix="", separator="_", ignored_keys=["innings"], special_keys=["players", "registry"]):
    flattened = {}
    
    def _flatten(obj, current_prefix):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ignored_keys:
                    continue
                    
                new_prefix = f"{current_prefix}{separator}{key}" if current_prefix else key
                
                if key == "players":
                    registry = obj.get('registry', {}).get('people', {})
                    # Get teams in the same order as in teams field
                    teams_order = obj.get('teams', list(value.keys()))
                    
                    for i, team in enumerate(teams_order, 1):
                        if team in value:
                            players = value[team]
                            # Store team name with team number
                            flattened[f"{new_prefix}{separator}team{i}"] = team
                            # Store player list with team number
                            flattened[f"{new_prefix}{separator}team{i}_list"] = str(players)
                            # Store player IDs with team number
                            for p, player in enumerate(players, 1):
                                player_id = registry.get(player, '')
                                flattened[f"{new_prefix}{separator}team{i}{separator}player{p}_id"] = player_id
                elif key == "registry":
                    if "people" in value:
                        flattened[f"{new_prefix}{separator}people_ids"] = str(list(value["people"].values()))
                elif isinstance(value, list):
                    flattened[new_prefix] = str(value)
                    for i, item in enumerate(value):
                        if isinstance(item, (dict, list)):
                            _flatten(item, f"{new_prefix}{separator}{i}")
                        else:
                            flattened[f"{new_prefix}{separator}{i}"] = item
                elif isinstance(value, (dict, list)):
                    _flatten(value, new_prefix)
                else:
                    flattened[new_prefix] = value
                    
    _flatten(data, prefix)
    return flattened

def get_new_matches(csv_path, final_dir):
    # Read existing CSV if it exists
    try:
        existing_df = pd.read_csv(csv_path)
        # Extract match IDs from filenames in the CSV if they exist
        existing_matches = set([f"{id}.json" for id in existing_df['match_id'].astype(str)] if 'match_id' in existing_df.columns else [])
    except :
        existing_matches = set()
        existing_df = None

    # Get all JSON files from directory
    all_files = [f for f in os.listdir(final_dir) if f.endswith('.json') and f != 'README.txt']
    
    # Find new matches
    new_matches = [f for f in all_files if f not in existing_matches]
    
    return new_matches, existing_df

def process_new_matches(new_matches, final_dir):
    all_matches_data = []
    
    for match_file in tqdm(new_matches):
        try:
            with open(os.path.join(final_dir, match_file), 'r', encoding='utf-8') as f:
                match_data = json.load(f)
                
            # Add match_id from filename
            match_id = match_file.split('.')[0]
            flattened_data = flatten_json(match_data)
            flattened_data['match_id'] = match_id
            
            all_matches_data.append(flattened_data)
        except Exception as e:
            print(f"Error processing {match_file}: {str(e)}")
            continue
    
    return pd.DataFrame(all_matches_data)

def update_matches_csv(csv_path, final_dir):
    # Get new matches
    new_matches, existing_df = get_new_matches(csv_path, final_dir)
    
    if not new_matches:
        print("No new matches to add.")
        return
    
    print(f"Found {len(new_matches)} new matches to process.")
    
    # Process new matches
    new_matches_df = process_new_matches(new_matches, final_dir)
    
    # Combine with existing data if it exists
    if existing_df is not None:
        # Ensure columns match
        missing_cols = set(new_matches_df.columns) - set(existing_df.columns)
        for col in missing_cols:
            existing_df[col] = None
            
        missing_cols = set(existing_df.columns) - set(new_matches_df.columns)
        for col in missing_cols:
            new_matches_df[col] = None
            
        # Combine dataframes
        final_df = pd.concat([existing_df, new_matches_df], ignore_index=True)
    else:
        final_df = new_matches_df
    
    # Save updated CSV
    final_df.to_csv(csv_path, index=False)
    print(f"Added {len(new_matches)} matches to the CSV.")
    print(f"Total matches in CSV: {len(final_df)}")

# Example usage
def add_data():
    update_matches_csv(paths['matches_csv'], paths['json_dir'])


def process_player_stats(json_data,match_id):
    # path = os.path.join(final_dir, )
    # Initialize defaultdict for each player's stats
    player_stats = defaultdict(lambda: defaultdict(int))
    
    # For tracking maiden overs
    current_over_runs = defaultdict(int)
    current_over_legit_balls = defaultdict(int)
    
    # Get basic match info
    match_info = json_data['info']
    teams = match_info['teams']
    registry = match_info['registry']['people']
    
    # Process each innings
    for innings in json_data['innings']:
        batting_team = innings['team']
        bowling_team = teams[1] if teams[0] == batting_team else teams[0]
        
        # Process each over
        for over in innings['overs']:
            # Reset over tracking for each new over
            current_over_runs.clear()
            current_over_legit_balls.clear()
            
            for delivery in over['deliveries']:
                batter = delivery['batter']
                bowler = delivery['bowler']
                non_striker = delivery['non_striker']
                runs = delivery.get('runs', {})
                extras = delivery.get('extras', {})
                
                # Track legitimate balls (not wide or no ball)
                is_extra = any(k in extras for k in ['wides', 'noballs', 'legbyes', 'byes'])
                is_legit_ball = not ('wides' in extras or 'noballs' in extras)
                
                # Batting stats
                if is_legit_ball:
                    player_stats[batter]['balls_faced'] += 1
                player_stats[batter]['runs_scored'] += runs.get('batter', 0)
                
                # Count boundaries
                if runs.get('batter', 0) == 4:
                    player_stats[batter]['fours'] += 1
                elif runs.get('batter', 0) == 6:
                    player_stats[batter]['sixes'] += 1
                
                # Bowling stats
                if is_legit_ball:
                    player_stats[bowler]['balls_bowled'] += 1
                    current_over_legit_balls[bowler] += 1
                
                # Track extra types
                for extra_type in ['wides', 'noballs', 'legbyes', 'byes']:
                    if extra_type in extras:
                        player_stats[bowler][f'extras_{extra_type}'] += extras[extra_type]
                
                # Track runs for maiden over calculation
                runs_in_over = runs.get('total', 0)
                if 'legbyes' in extras:
                    runs_in_over -= extras['legbyes']
                if 'byes' in extras:
                    runs_in_over -= extras['byes']
                current_over_runs[bowler] += runs_in_over
                player_stats[bowler]['runs_conceded'] += runs_in_over
                
                # Process wickets
                if 'wickets' in delivery:
                    for wicket in delivery['wickets']:
                        player_out = wicket['player_out']
                        kind = wicket['kind']
                        
                        # Update bowler wickets
                        if kind not in ['run out']:
                            player_stats[bowler]['wickets'] += 1
                            if kind in ['bowled', 'lbw']:
                                player_stats[bowler]['wicket_bowled_lbw'] += 1
                        
                        # Update fielding stats
                        if 'fielders' in wicket:
                            for fielder in wicket['fielders']:
                                fielder_name = fielder if isinstance(fielder, str) else fielder.get('name')
                                if kind == 'caught':
                                    player_stats[fielder_name]['catches'] += 1
                                elif kind == 'stumped':
                                    player_stats[fielder_name]['stumpings'] += 1
                                elif kind == 'run out':
                                    # In cricsheet, if there's only one fielder listed, it's a direct hit
                                    if len(wicket['fielders']) == 1:
                                        player_stats[fielder_name]['run_outs_direct'] += 1
                                    else:
                                        player_stats[fielder_name]['run_outs_indirect'] += 1
                                        
                        # Update duck for batter
                        if player_stats[player_out]['runs_scored'] == 0:
                            player_stats[player_out]['duck'] = 1
            
            # Check for maiden over at end of over
            for bowler in current_over_runs:
                if current_over_runs[bowler] == 0 and current_over_legit_balls[bowler] == 6:
                    player_stats[bowler]['maiden_overs'] += 1
    
    # Create DataFrame rows
    rows = []
    for team_idx, team in enumerate(teams, 1):
        for player in match_info['players'][team]:
            player_id = registry[player]
            stats = player_stats[player]
            
            # Calculate overs bowled
            balls_bowled = stats['balls_bowled']
            overs = balls_bowled // 6
            remaining_balls = balls_bowled % 6
            overs_bowled = f"{overs}.{remaining_balls}"
            
            # Calculate strike rate and economy
            strike_rate = (stats['runs_scored'] / stats['balls_faced'] * 100) if stats['balls_faced'] > 0 else 0
            economy = (stats['runs_conceded'] / (stats['balls_bowled']/6)) if stats['balls_bowled'] > 0 else 0
            
            row = {
                'match_id': match_id,
                # 'date': match_info['dates'][0],
                'player_id': player_id,
                'player_name': player,
                'team': team,
                'team_no': team_idx,
                
                # Batting stats
                'runs_scored': stats['runs_scored'],
                'balls_faced': stats['balls_faced'],
                'fours': stats['fours'],
                'sixes': stats['sixes'],
                'strike_rate': strike_rate,
                'duck': stats['duck'],
                
                # Bowling stats
                'overs_bowled': overs_bowled,
                'balls_bowled': stats['balls_bowled'],  # Only legitimate balls
                'runs_conceded': stats['runs_conceded'],
                'wickets': stats['wickets'],
                'wicket_bowled_lbw': stats['wicket_bowled_lbw'],
                'maiden_overs': stats['maiden_overs'],
                'economy': economy,
                'extras_wides': stats['extras_wides'],
                'extras_noballs': stats['extras_noballs'],
                'extras_legbyes': stats['extras_legbyes'],
                'extras_byes': stats['extras_byes'],
                
                # Fielding stats
                'catches': stats['catches'],
                'stumpings': stats['stumpings'],
                'run_outs_direct': stats['run_outs_direct'],
                'run_outs_indirect': stats['run_outs_indirect']
            }
            rows.append(row)
            
    return pd.DataFrame(rows)

def process_match_stats(matches_df_path,final_dir,player_stats_path):
    # Read or create matches DataFrame
    try:
        matches_df = pd.read_csv(matches_df_path)
        matches_df['match_id'] = matches_df['match_id'].apply(lambda x: str(x))
        if 'processed' not in matches_df.columns:
            matches_df['processed'] = False
    except FileNotFoundError:
        print("Matches DataFrame not found. Please create it first.")
        return
    
    # Read or create player stats DataFrame
    try:
        player_stats_df = pd.read_csv(player_stats_path)
        
    except :
        player_stats_df = pd.DataFrame()
    
    print(len(matches_df))
    # Get unprocessed matches
    matches_df['match_id']=matches_df['match_id'].astype('str')
    print('ok')
    print(player_stats_df.info())
    player_stats_df['match_id']=player_stats_df['match_id'].astype('str')
    s11 = set(player_stats_df['match_id'])
    unprocessed_matches = matches_df[~(matches_df['match_id'].isin(s11))]
    
    if len(unprocessed_matches) == 0:
        print("No new matches to process.")
        return
    
    print(f"Found {len(unprocessed_matches)} unprocessed matches.")
    
    # Process each unprocessed match
    new_stats_dfs = []
    processed_match_ids = []
    
    for idx, match in tqdm(unprocessed_matches.iterrows()):
        match_id = match['match_id']
        json_file = f"{match_id}.json"
        json_path = os.path.join(final_dir, json_file)
        
        try:
            # Read and process match JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Process player stats for this match
            
            match_stats_df = process_player_stats(match_data,match_id)
            new_stats_dfs.append(match_stats_df)
            processed_match_ids.append(match_id)
            # print(f"Processed match {match_id}")
            
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")
            continue
    
    # Update matches DataFrame
    matches_df.loc[matches_df['match_id'].isin(processed_match_ids), 'processed'] = True
    print(len(matches_df))
    # Combine and save player stats
    if new_stats_dfs:
        new_stats_combined = pd.concat(new_stats_dfs, ignore_index=True)
        if not player_stats_df.empty:
            player_stats_df = pd.concat([player_stats_df, new_stats_combined], ignore_index=True)
        else:
            player_stats_df = new_stats_combined
    player_stats_df = player_stats_df.drop_duplicates()
    player_stats_df['match_id'] = player_stats_df['match_id'].apply(lambda x: str(x))
    set1 = set(matches_df[matches_df['processed']==1]['match_id'])
    print(len(set1))
    print(matches_df['processed'].dtype,matches_df['processed'].value_counts())
    set2 = set(player_stats_df['match_id'])
    print(len(set2))
    print(len(set1-set2),len(set2-set1))
    matches_df['processed']= matches_df['match_id'].apply(lambda x: True if x in (set2) else False)
    set1 = set(matches_df[matches_df['processed']==1]['match_id'])
    player_stats_df = player_stats_df[player_stats_df['match_id'].apply(lambda x: True if x in (set1) else False)]
    set2 = set(player_stats_df['match_id'])
    assert set1==set2
    matches_df.to_csv(matches_df_path, index=False)
    player_stats_df.to_csv(player_stats_path, index=False)
    print(f"Added stats for {len(processed_match_ids)} matches to player stats DataFrame")
    print(f"Total matches in player stats: {len(player_stats_df['match_id'].unique())}")


# Example usage
def update_data():
    process_match_stats(paths['matches_csv'], paths['json_dir'], paths['player_stats_csv'])


def calculate_fantasy_points(paths,paths_pro):
    """
    Calculate format-specific fantasy points for each player
    """
    matches_df = pd.read_csv(paths['matches_csv'],low_memory=False)
    matches_df.to_csv(paths_pro['matches_csv'],index=False)


    matches_df['match_id']= matches_df['match_id'].apply(lambda x:str(x))
    # print(len(matches_df))
    matches_df = matches_df[matches_df['processed']!=0]
    matches_df = matches_df.reset_index(drop=True)
    # print(len(matches_df))
    # print(matches_df.columns)
    matches_df.set_index('match_id', inplace=True)
    player_stats_df = pd.read_csv(paths['player_stats_csv'])
    player_stats_df['match_id'] = player_stats_df['match_id'].apply(lambda x:str(x))

    # Create a copy to avoid modifying original
    df = player_stats_df.copy()
    # set1 = set(matches_df.index)
    # set2 = set(player_stats_df['match_id'])
    # print(set1-set2)
    # Add match format to player stats
    df['format'] = matches_df.loc[df['match_id'], 'info_match_type'].values
    
    # Define format-specific scoring rules
    scoring_rules = {
        'T20': {
            'min_overs_eco': 2,
            'min_balls_sr': 10,
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'thirty_bonus': 4,
            'fifty_bonus': 8,
            'hundred_bonus': 16,
            'duck_penalty': -2,
            'wicket_point': 25,
            'lbw_bowled_bonus': 8,
            'maiden_over': 12,
            'three_wicket_bonus': 4,
            'four_wicket_bonus': 8,
            'five_wicket_bonus': 16,
            'catch_point': 8,
            'three_catch_bonus': 4,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6,
            'eco_ranges': [(0, 5, 6), (5, 6, 4), (6, 7, 2), 
                          (10, 11, -2), (11, 12, -4), (12, float('inf'), -6)],
            'sr_ranges': [(170, float('inf'), 6), (150, 170, 4), (130, 150, 2),
                         (60, 70, -2), (50, 60, -4), (0, 50, -6)]
        },
        'MDM': {
            'min_overs_eco': 2,
            'min_balls_sr': 10,
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'thirty_bonus': 4,
            'fifty_bonus': 8,
            'hundred_bonus': 16,
            'duck_penalty': -2,
            'wicket_point': 25,
            'lbw_bowled_bonus': 8,
            'maiden_over': 12,
            'three_wicket_bonus': 4,
            'four_wicket_bonus': 8,
            'five_wicket_bonus': 16,
            'catch_point': 8,
            'three_catch_bonus': 4,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6,
            'eco_ranges': [(0, 5, 6), (5, 6, 4), (6, 7, 2), 
                          (10, 11, -2), (11, 12, -4), (12, float('inf'), -6)],
            'sr_ranges': [(170, float('inf'), 6), (150, 170, 4), (130, 150, 2),
                         (60, 70, -2), (50, 60, -4), (0, 50, -6)]
        },
        'IT20': {
            'min_overs_eco': 2,
            'min_balls_sr': 10,
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'thirty_bonus': 4,
            'fifty_bonus': 8,
            'hundred_bonus': 16,
            'duck_penalty': -2,
            'wicket_point': 25,
            'lbw_bowled_bonus': 8,
            'maiden_over': 12,
            'three_wicket_bonus': 4,
            'four_wicket_bonus': 8,
            'five_wicket_bonus': 16,
            'catch_point': 8,
            'three_catch_bonus': 4,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6,
            'eco_ranges': [(0, 5, 6), (5, 6, 4), (6, 7, 2), 
                          (10, 11, -2), (11, 12, -4), (12, float('inf'), -6)],
            'sr_ranges': [(170, float('inf'), 6), (150, 170, 4), (130, 150, 2),
                         (60, 70, -2), (50, 60, -4), (0, 50, -6)]
        },
        'ODI': {
            'min_overs_eco': 5,
            'min_balls_sr': 20,
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'fifty_bonus': 4,
            'hundred_bonus': 8,
            'duck_penalty': -3,
            'wicket_point': 25,
            'lbw_bowled_bonus': 8,
            'maiden_over': 4,
            'four_wicket_bonus': 4,
            'five_wicket_bonus': 8,
            'catch_point': 8,
            'three_catch_bonus': 4,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6,
            'eco_ranges': [(0, 2.5, 6), (2.5, 3.5, 4), (3.5, 4.5, 2),
                          (7, 8, -2), (8, 9, -4), (9, float('inf'), -6)],
            'sr_ranges': [(140, float('inf'), 6), (120, 140, 4), (100, 120, 2),
                         (40, 50, -2), (30, 40, -4), (0, 30, -6)]
        },
        'ODM': {
            'min_overs_eco': 5,
            'min_balls_sr': 20,
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'fifty_bonus': 4,
            'hundred_bonus': 8,
            'duck_penalty': -3,
            'wicket_point': 25,
            'lbw_bowled_bonus': 8,
            'maiden_over': 4,
            'four_wicket_bonus': 4,
            'five_wicket_bonus': 8,
            'catch_point': 8,
            'three_catch_bonus': 4,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6,
            'eco_ranges': [(0, 2.5, 6), (2.5, 3.5, 4), (3.5, 4.5, 2),
                          (7, 8, -2), (8, 9, -4), (9, float('inf'), -6)],
            'sr_ranges': [(140, float('inf'), 6), (120, 140, 4), (100, 120, 2),
                         (40, 50, -2), (30, 40, -4), (0, 30, -6)]
        },
        'Test': {
            'run_point': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'fifty_bonus': 4,
            'hundred_bonus': 8,
            'duck_penalty': -4,
            'wicket_point': 16,
            'lbw_bowled_bonus': 8,
            'four_wicket_bonus': 4,
            'five_wicket_bonus': 8,
            'catch_point': 8,
            'stumping_point': 12,
            'runout_direct': 12,
            'runout_indirect': 6
        }
    }
    
    # Initialize fantasy points
    df['fantasy_points'] = 0
    
    # Calculate points for each format
    for format_type in scoring_rules.keys():
        format_mask = df['format'] == format_type
        rules = scoring_rules[format_type]
        
        # Basic batting points
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'runs_scored'] * rules['run_point']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'fours'] * rules['boundary_bonus']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'sixes'] * rules['six_bonus']
        
        # Milestone batting points
        if format_type in (['T20','MDM','IT20']):
            df.loc[format_mask & (df['runs_scored'] >= 30) & (df['runs_scored'] < 50), 'fantasy_points'] += rules['thirty_bonus']
        df.loc[format_mask & (df['runs_scored'] >= 50) & (df['runs_scored'] < 100), 'fantasy_points'] += rules['fifty_bonus']
        if 'hundred_bonus' in rules:
            df.loc[format_mask & (df['runs_scored'] >= 100), 'fantasy_points'] += rules['hundred_bonus']
        
        # Duck penalty
        df.loc[format_mask & df['duck'].astype(bool), 'fantasy_points'] += rules['duck_penalty']
        
        # Bowling points
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'wickets'] * rules['wicket_point']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'wicket_bowled_lbw'] * rules['lbw_bowled_bonus']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'maiden_overs'] * rules.get('maiden_over', 0)
        
        # Bowling milestone points
        if 'three_wicket_bonus' in rules:
            df.loc[format_mask & (df['wickets'] >= 3), 'fantasy_points'] += rules['three_wicket_bonus']
        if 'four_wicket_bonus' in rules:
            df.loc[format_mask & (df['wickets'] >= 4), 'fantasy_points'] += rules['four_wicket_bonus']
        if 'five_wicket_bonus' in rules:
            df.loc[format_mask & (df['wickets'] >= 5), 'fantasy_points'] += rules['five_wicket_bonus']
        
        # Fielding points
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'catches'] * rules['catch_point']
        if 'three_catch_bonus' in rules:
            df.loc[format_mask & (df['catches'] >= 3), 'fantasy_points'] += rules['three_catch_bonus']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'stumpings'] * rules['stumping_point']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'run_outs_direct'] * rules['runout_direct']
        df.loc[format_mask, 'fantasy_points'] += df.loc[format_mask, 'run_outs_indirect'] * rules['runout_indirect']
        
        # Economy rate points (only for limited overs)
        if 'eco_ranges' in rules:
            overs_bowled = df.loc[format_mask, 'balls_bowled'] / 6
            economy_rate = (df.loc[format_mask, 'runs_conceded'] / overs_bowled).fillna(0)
            eco_points = pd.Series(0, index=df.loc[format_mask].index)
            
            min_overs = rules.get('min_overs_eco', 0)
            mask = overs_bowled >= min_overs
            
            for low, high, points in rules['eco_ranges']:
                eco_points[mask & (economy_rate >= low) & (economy_rate < high)] = points
                
            df.loc[format_mask, 'fantasy_points'] += eco_points
        
        # Strike rate points (only for limited overs)
        if 'sr_ranges' in rules:
            sr = (df.loc[format_mask, 'runs_scored'] / df.loc[format_mask, 'balls_faced'] * 100).fillna(0)
            sr_points = pd.Series(0, index=df.loc[format_mask].index)
            
            min_balls = rules.get('min_balls_sr', 0)
            mask = df.loc[format_mask, 'balls_faced'] >= min_balls
            
            for low, high, points in rules['sr_ranges']:
                sr_points[mask & (sr > low) & (sr <= high)] = points
                
            df.loc[format_mask, 'fantasy_points'] += sr_points
    df.to_csv(paths_pro['player_stats_csv'],index=False)
    
    return df



def get_roles():
    # try:
    #     peeps = pd.read_csv('https://cricsheet.org/register/people.csv')
    # except:
    #     peeps = pd.read_csv('data/raw/cricksheet_data/people.csv')
    li = pd.read_csv('src/data/raw/additional_data/available_roles.csv')
#     peeps = peeps.dropna(subset='key_cricinfo').reset_index(drop=True)
#     peeps['player_id']=peeps['identifier']
#     set_p = set(li['player_id'])
#     # merged = peeps.merge(li[['player_id','f_pos']], on='player_id', how='left')
#     peeps_sub = peeps[~peeps['player_id'].isin(set_p)].reset_index(drop=True)
#     class Player:
#         def __init__(self, player_id, session):
#             self.player_id = player_id
#             self.url = f"https://www.espncricinfo.com/player/player-name-{player_id}"
#             self.json_url = f"http://core.espnuk.org/v2/sports/cricket/athletes/{player_id}"
#             self.headers = {'user-agent': 'Mozilla/5.0'}
#             self.session = session
#             self.json = None
#             self.playing_role = None

#         async def get_html(self):
#             async with self.session.get(self.url, headers=self.headers) as response:
#                 if response.status == 404:
#                     raise PlayerNotFoundError
#                 return BeautifulSoup(await response.text(), 'html.parser')

#         async def get_json(self):
#             async with self.session.get(self.json_url, headers=self.headers) as response:
#                 if response.status == 404:
#                     raise PlayerNotFoundError
#                 return await response.json()

#         async def initialize(self):
#             self.json = await self.get_json()
#             self.playing_role = self._playing_role()
#             self.dob = self.json['dateOfBirth']
#             self.fullname= self.json['fullName']
#             self.firstname = self.json['firstName']
#             self.lastname = self.json['lastName']

#         def _playing_role(self):
#             # Logic to extract the playing role from self.json
#             return {"name": self.json.get("position")}

#     async def fetch_player_data(row, session):
#         try:
#             p = Player(str(int(row['key_cricinfo'])), session)
#             await p.initialize()
#             return row['identifier'], {"role": p.playing_role['name'],'fname':p.firstname,'lname':p.lastname,'name':p.fullname, "player_object": p}
#         except:
#             try:
#                 p = Player(str(int(row['key_cricinfo_2'])), session)
#                 await p.initialize()
#                 return row['identifier'], {"role": p.playing_role['name'],'fname':p.firstname,'lname':p.lastname,'name':p.fullname, "player_object": p}
#             except Exception as e:
#                 return row['identifier'], {"error": e, "data": row}

#     async def main(peeps):
#         p_roles = {}
#         errors = []

#         async with aiohttp.ClientSession() as session:
#             tasks = [fetch_player_data(row, session) for _, row in peeps.iterrows()]
#             for result in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
#                 identifier, data = await result
#                 if "error" in data:
#                     errors.append(data)
#                 else:
#                     p_roles[identifier] = data

#         return p_roles, errors
#     nest_asyncio.apply()

# # Run the async function
#     results = asyncio.run(main(peeps_sub))
#     p_roles=results[0]
#     li_l = [{'player_id':id, 'role': v['role']['name'],'fname':v['fname'],'lname':v['lname'],'name':v['name']} for id,v in p_roles.items()]
#     li_l = pd.DataFrame(li_l)
#     role_mapping = {
#         "Bowler": "Bowler",
#         "Allrounder": "Allrounder",
#         "Bowling allrounder": "Allrounder",
#         "Batting allrounder": "Allrounder",
#         "Wicketkeeper batter": "Wicketkeeper",
#         "Wicketkeeper": "Wicketkeeper",
#         "Batter": "Batter",
#         "Middle-order batter": "Batter",
#         "Top-order batter": "Batter",
#         "Opening batter": "Batter",
#     }
#     li_l["mapped_role"] = li_l["role"].map(role_mapping).fillna(np.nan)
#     li = li.merge(li_l[['mapped_role','name','player_id']], on='player_id', how='left', suffixes=('', '_df2'))
#     li['mapped_role'] = li['mapped_role'].fillna(li['mapped_role_df2'])
#     li['name'] = li['name'].fillna(li['name_df2'])
#     li['f_pos']=li['f_pos'].where(li['f_pos'].notna(),li['mapped_role'])
#     li = li.drop(columns=['mapped_role_df2','name_df2'])
    
#     li.to_csv('data/raw/additional_data/available_roles.csv',index=False)
    return li

def ensure_roles(group):
    missing_roles = required_roles - set(group['f_pos'].dropna())
    
    if missing_roles:
        # print(missing_roles)
        # Assign missing roles to players with NaN f_pos first
        available_players = group[group['f_pos'].isna()]
        for role, player_index in zip(missing_roles, available_players.index):
            group.loc[player_index, 'f_pos'] = role

        # If there are still missing roles, assign them arbitrarily
        remaining_roles = missing_roles - set(group.loc[group['f_pos'].notna(), 'f_pos'])
        available_players = group.index[group['f_pos']==group['f_pos'].value_counts().index[0]]
        # print(available_players)
        for i,role in enumerate(remaining_roles):
            group.loc[available_players[i], 'f_pos'] = role  # Assign to first player arbitrarily
    return group


def get_data():
    # load_data()
    add_data()
    update_data()
    player_stats_df = calculate_fantasy_points(paths, paths_pro)
    m_df = pd.read_csv(paths_pro['matches_csv'],low_memory=0)
    player_stats_df=player_stats_df.merge(m_df[['match_id','info_dates_0']],how='left',on='match_id')
    player_stats_df['date'] = pd.to_datetime(player_stats_df['info_dates_0'])
    player_stats_df = player_stats_df.drop(columns=['info_dates_0'])
    a_r = get_roles()
    players_with_stumpings = set(player_stats_df[player_stats_df['stumpings'] > 0]['player_id'])
    a_r.loc[a_r['player_id'].isin(players_with_stumpings), 'f_pos'] = 'Wicketkeeper'
    players_with_overs = set(player_stats_df[player_stats_df['overs_bowled'] > 0]['player_id'])
    players_with_runs = set(player_stats_df[player_stats_df['runs_scored'] > 0]['player_id'])
    players_with_both = players_with_overs|players_with_runs
    a_r.loc[a_r['player_id'].isin(players_with_both) & a_r['f_pos'].isna(), 'f_pos'] = 'Allrounder'
    a_r.loc[a_r['player_id'].isin(players_with_overs) & a_r['f_pos'].isna(), 'f_pos'] = 'Bowler'
    a_r.loc[a_r['player_id'].isin(players_with_runs) & a_r['f_pos'].isna(), 'f_pos'] = 'Batter'
    players_with_catches = set(player_stats_df[player_stats_df['catches'] > 0]['player_id'])
    a_r.loc[a_r['player_id'].isin(players_with_catches) & a_r['f_pos'].isna(), 'f_pos'] = 'Wicketkeeper'
    ppp  = pd.merge(player_stats_df.drop(columns=['f_pos']),a_r[['player_id','f_pos']],how='left',on='player_id')
    ppp = ppp.groupby('match_id', group_keys=False).apply(ensure_roles)
    ppp.to_csv(paths_pro['player_stats_csv'],index=False)
    a_r.to_csv(paths_pro['roles'],index=False)

get_data()