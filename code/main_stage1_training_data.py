# File name: main_stage1_training_data.py
# Description: Process streaming history and track features into 
#              objects suitable for model training and save as
#              .csv files, including constructing stage 1 outcomes:
#              playlist and likeability score
# Author: Chris Rowe

import spotify_modeling as sm
import os
import json
import pandas as pd
from config import *

def main():

    # navigate to root directory
    os.chdir(os.path.dirname(os.getcwd()))

    # import processed streamining data
    df = pd.read_csv('data/processed/track_features.csv')
    df_history = pd.read_csv('data/processed/streaming_history.csv')

    # generate outcomes
    print("Generating and exporting outcomes...")

    # import raw playlist data
    with open('data/raw/Playlist1.json', 'r') as f:
        playlist_raw = json.load(f)

    # obtain playlist outcome
    playlist_tracks = sm.getPlaylistOutcome(playlist_raw, good_playlist_names)

    # organize training data
    X, X_columns, y_score, y_playlist = sm.getTrainingData(df_history, df, playlist_tracks)

    # convert to dataframes
    X_df = pd.DataFrame(X, columns = X_columns)
    y_score = pd.DataFrame(y_score, columns = ['y_score'])
    y_playlist = pd.DataFrame(y_playlist, columns = ['y_playlist'])

    # save training data as csv
    X_df.to_csv('data/processed/X.csv', index=False)
    y_score.to_csv('data/processed/y_score.csv', index=False)
    y_playlist.to_csv('data/processed/y_playlist.csv', index=False)
    print("--Outcomes generated and exported!")

if __name__ == '__main__':
    main()

