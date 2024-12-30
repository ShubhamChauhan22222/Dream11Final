
import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import numpy as np
import lightgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from math import floor
from sklearn.base import BaseEstimator, RegressorMixin

import sys 
sys.path.append('src/data_processing/')
sys.path.append('src/model/')

import data_download
import feature_engineering
import train_model
import predict_model



model_path= None
uploaded_file = 'src/data/processed/full_data_features.csv'
# full_data = pd.read_csv(uploaded_file, low_memory=False)

source_directory = "src/out_of_sample_data"
destination_directory = "src/data/raw/cricksheet_data/json_files"

import os
import shutil

def move_files(source_dir, destination_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    os.makedirs(destination_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        try:
            shutil.move(source_path, destination_path)
            # print(f"Moved: {source_path} -> {destination_path}")
        except Exception as e:
            print(f"Error moving {source_path}: {e}")
    print('Moving Done!')


def lets_get_data():
    is_empty = lambda path: os.path.isdir(source_directory) and not any(os.scandir(source_directory))
    if is_empty:
        data_download.load_data()
    else:
        move_files(source_directory, destination_directory)
    data_download.get_data()
    return feature_engineering.get_features()






# Set page title
st.title("Model Evaluation and Training Interface")

# Inputs for training and testing periods
st.sidebar.header("Specify Training and Testing Periods")
training_start = st.sidebar.date_input("Training Start Date", datetime(2010, 1, 1))
training_end = st.sidebar.date_input("Training End Date", datetime(2024, 5, 30))
testing_start = st.sidebar.date_input("Testing Start Date", datetime(2024, 8, 1))
testing_end = st.sidebar.date_input("Testing End Date", datetime(2024, 9, 2))

# Initialize session state for expander visibility
if "show_expander" not in st.session_state:
    st.session_state["show_expander"] = False

def split_data(df, train_start_date, train_end_date, test_start_date, test_end_date):

    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values(by='match_date')

    # Convert string inputs to datetime for comparison
    train_start_date = pd.to_datetime(train_start_date)
    train_end_date = pd.to_datetime(train_end_date)
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = pd.to_datetime(test_end_date)

    train_df = df[(df['match_date'] >= train_start_date) & (df['match_date'] <= train_end_date)].copy()

    # Filter data for testing
    test_df = df[(df['match_date'] >= test_start_date) & (df['match_date'] <= test_end_date)].copy()

    return train_df, test_df

# import pandas as pd

def select_dream_team(df, column):
    roles_required = ['Allrounder', 'Bowler', 'Batter', 'Wicketkeeper']
    teams_required = set(df['team'].unique())

    selected_players = set()  # To store unique player IDs
    role_filled = set()  # To track fulfilled roles
    team_filled = set()  # To track fulfilled teams

    # Step 1: Select the best player for each role
    for role in roles_required:
        role_players = df[df['f_pos'] == role].sort_values(by=column, ascending=False)
        if not role_players.empty:
            best_player = role_players.iloc[0]
            selected_players.add(best_player['player_id'])
            role_filled.add(role)
            team_filled.add(best_player['team'])

    # Step 2: Select the best player for each team
    for team in teams_required:
        if team not in team_filled:
            team_players = df[df['team'] == team].sort_values(by=column, ascending=False)
            if not team_players.empty:
                best_player = team_players.iloc[0]
                selected_players.add(best_player['player_id'])
                team_filled.add(team)

    # Step 3: Fill remaining roles and teams
    remaining_roles = set(roles_required) - role_filled
    remaining_teams = teams_required - team_filled

    for role in remaining_roles:
        role_players = df[df['f_pos'] == role].sort_values(by=column, ascending=False)
        for _, player in role_players.iterrows():
            if player['player_id'] not in selected_players:
                selected_players.add(player['player_id'])
                team_filled.add(player['team'])
                break

    for team in remaining_teams:
        team_players = df[df['team'] == team].sort_values(by=column, ascending=False)
        for _, player in team_players.iterrows():
            if player['player_id'] not in selected_players:
                selected_players.add(player['player_id'])
                break

    # Step 4: Fill remaining slots to make up 11 players
    remaining_players = df[~df['player_id'].isin(selected_players)].sort_values(by=column, ascending=False)
    for _, player in remaining_players.iterrows():
        if len(selected_players) < 11:
            selected_players.add(player['player_id'])
        else:
            break

    # Step 5: Return the selected team
    return df[df['player_id'].isin(selected_players)]


# import pandas as pd

def prepare_summary_row(actual_filtered, predicted_filtered, team1, team2, date):
    match_date = date
    predicted_filtered['predictions'] = np.ceil(predicted_filtered['predictions'])
    # Prepare predicted player data
    # predicted_filtered = predicted_filtered.reset_index(drop=True)
    # predicted_filtered = predicted_filtered.sort_values(by='predictions')
    # predicted_filtered.loc[0,'predictions'] = predicted_filtered.loc[0,'predictions']*2
    # predicted_filtered.loc[1,'predictions'] = predicted_filtered.loc[1,'predictions']*1.5
    predicted_players = predicted_filtered[['name', 'predictions']].values
    predicted_player_data = {}
    for i, (player, points) in enumerate(predicted_players, start=1):
        predicted_player_data[f'Predicted Player {i}'] = player
        predicted_player_data[f'Predicted Player {i} Points'] = points

    # Prepare actual player data
    actual_filtered['target'] = np.ceil(actual_filtered['target'])
    # actual_filtered = actual_filtered.reset_index(drop=True)
    # actual_filtered = actual_filtered.sort_values(by='target')
    # actual_filtered.loc[0,'target'] = actual_filtered.loc[0,'target']*2
    # actual_filtered.loc[1,'target'] = actual_filtered.loc[1,'target']*1.5
    actual_players = actual_filtered[['name', 'target']].values
    actual_player_data = {}
    for i, (player, points) in enumerate(actual_players, start=1):
        actual_player_data[f'Dream Team Player {i}'] = player
        actual_player_data[f'Dream Team Player {i} Points'] = points

    # Calculate total points and MAE
    total_predicted_points = predicted_filtered['predictions'].sum()
    total_actual_points = actual_filtered['target'].sum()
    mae = abs(total_predicted_points - total_actual_points)

    # Combine all data into a single row
    summary_data = {
        'Match Date': match_date,
        'Team 1': team1,
        'Team 2': team2,
        **predicted_player_data,
        **actual_player_data,
        'Total Predicted Points': total_predicted_points,
        'Total Dream Team Points': total_actual_points,
        'Total Points MAE': mae
    }
    try:
        assert len(pd.DataFrame([summary_data]).iloc[0]) == 50
    except:
        print(predicted_players)
        assert 0
    return pd.DataFrame([summary_data])

def prepare_summary_row_with_captain_vc(actual_filtered, predicted_filtered, team1, team2, date):
    match_date = date
    predicted_filtered['predictions'] = np.ceil(predicted_filtered['predictions'])

    # Prepare predicted player data
    predicted_players = predicted_filtered[['name', 'predictions']].values
    predicted_player_data = {}
    for i, (player, points) in enumerate(predicted_players, start=1):
        predicted_player_data[f'Predicted Player {i}'] = player
        predicted_player_data[f'Predicted Player {i} Points'] = points

    # Prepare actual player data
    actual_filtered['target'] = np.ceil(actual_filtered['target'])
    actual_players = actual_filtered[['name', 'target']].values
    actual_player_data = {}
    for i, (player, points) in enumerate(actual_players, start=1):
        actual_player_data[f'Dream Team Player {i}'] = player
        actual_player_data[f'Dream Team Player {i} Points'] = points

    # Calculate captain and vice-captain adjusted points
    predicted_sorted = predicted_filtered.sort_values(by='predictions', ascending=False)
    actual_sorted = actual_filtered.sort_values(by='target', ascending=False)

    predicted_points = predicted_sorted['predictions'].values
    actual_points = actual_sorted['target'].values

    total_predicted_points = (predicted_points[0] * 2) + (predicted_points[1] * 1.5) + predicted_points[2:].sum()
    total_actual_points = (actual_points[0] * 2) + (actual_points[1] * 1.5) + actual_points[2:].sum()

    # Calculate MAE
    mae = abs(total_predicted_points - total_actual_points)

    # Combine all data into a single row
    summary_data = {
        'Match Date': match_date,
        'Team 1': team1,
        'Team 2': team2,
        **predicted_player_data,
        **actual_player_data,
        'Total Predicted Points': total_predicted_points,
        'Total Dream Team Points': total_actual_points,
        'Total Points MAE': mae
    }

    try:
        assert len(pd.DataFrame([summary_data]).iloc[0]) == 50
    except AssertionError:
        print(predicted_players)
        raise

    return pd.DataFrame([summary_data])

def get_csv_per_match(df):
    df = df.copy()
    df1 = select_dream_team(df,'predictions')
    try:
        assert len(df1) == 11
    except:
        print(df)
        assert 0
    df2 = select_dream_team(df, 'target')
    assert len(df2) == 11
    teams = list(set(df['team']))
    date = df['match_date'].iloc[0]
    return prepare_summary_row(df2, df1, teams[0], teams[1], date)

def get_csv( f_test,y_pred_test):
    f_test = f_test.copy()
    f_test['predictions'] = y_pred_test
    final_csv = f_test.groupby('match_id').apply(get_csv_per_match)
    return final_csv



def train_modelll():
    global model_path
    st.session_state["show_expander"] = True

    
    f_train, f_test = split_data(full_data, training_start, training_end, testing_start, testing_end)
    f_train.to_csv(f'src/data/processed/training_data_{training_end}',index=False)
    X = f_train.drop(['target', 'player_id', 'match_id', 'match_date', 'name', 'team'], axis=1)
    X_test = f_test.drop(['target', 'player_id', 'match_id', 'match_date', 'name', 'team'], axis=1)
    X['format'] = X['format'].astype('category')
    X_test['format'] = X_test['format'].astype('category')
    X['info_gender'] = X['info_gender'].astype('category')
    X_test['info_gender'] = X_test['info_gender'].astype('category')
    X['f_pos'] = X['f_pos'].astype('category')
    X_test['f_pos'] = X_test['f_pos'].astype('category')
    y = f_train['target']
    y_test = f_test['target']

    # Initialize and train HybridModel
    model = train_model.train(X,y)

    # Predict and evaluate
    y_pred_test = predict_model.predict(X_test,model)
    # Generate predictions for testing period
    predictions = get_csv(f_test, y_pred_test)
    model_path = f"src/model_artifacts/model_{training_end}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model}, f)

    # Store predictions and MAE in session state
    st.session_state["f_train"] = f_train
    st.session_state["predictions"] = predictions
    st.session_state["overall_mae"] = predictions['Total Points MAE'].mean()  # Example value
    st.session_state["overall_MAPE"] = (predictions['Total Points MAE']/predictions['Total Dream Team Points']).mean()*100 # Example value


# Train model button
if st.button("Train Model"):
    st.write("### Fetching the data......")
    full_data = lets_get_data()
    st.write("### Training the model on the specified period...")
    train_modelll()

# Display training details in expander if session state indicates so
if st.session_state["show_expander"]:
    with st.expander("Training Details", expanded=True):

        # Retrieve predictions from session state
        predictions = st.session_state.get("predictions", pd.DataFrame())

        st.write("#### Predictions:")
        st.dataframe(predictions, use_container_width=True)

        # Provide download link for train_CSV
        f_train = st.session_state.get("f_train", pd.DataFrame())
        f_train = f_train.to_csv(index=False).encode('utf-8')
        st.write("#### Download Training_data:")
        st.download_button(
            label="Download Training data CSV",
            data=f_train,
            file_name=f"Training_data_{training_end}.csv",
            mime="text/csv",
        )

        # Provide download link for predictions CSV
        st.write("#### Download Predictions:")
        csv_file = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv_file,
            file_name="predictions_testing_period.csv",
            mime="text/csv",
        )

        st.write("#### Download Trained Model:")
        with open(model_path, "rb") as f:
            st.download_button(
                label="Download Trained Model",
                data=f,
                file_name=f"model_{training_end}.pkl",
            )

        # Display the overall MAE
        overall_mae = st.session_state.get("overall_mae", 0)
        st.metric(label="Overall Mean Absolute Error (MAE)", value=floor(overall_mae))

        # Display the overall MAPE
        overall_MAPE = st.session_state.get("overall_MAPE", 0)
        st.write("#### Model Performance:")
        st.metric(label="Overall Mean Absolute Percentage Error (MAPE)", value=floor(overall_MAPE))


