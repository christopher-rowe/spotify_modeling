#!/usr/bin/env python3
# File name: main.py
# Description: Obtain features for all tracks in streaming 
#              history, generate outcomes for model fitting
#              and export all as .csv files
# Author: Chris Rowe
# Date: 04-07-2020

import spotify_modeling as sm
import os
import json
import pandas as pd
import ast
from config import *

def main():

    # get spotify authentication token
    print("Getting the spotify authentication token...")
    auth, token,  refresh_token= sm.get_token(username, client_id, 
                                              client_secret, redirect_uri, scope)
    print("--Token recieved!")
   
    # convert streaming data from .json to python dictionary
    print("Loading raw streaming data...")
    os.chdir(os.path.dirname(os.getcwd()))
    with open('data/raw/StreamingHistory0.json', 'r', encoding='UTF-8') as f:
        streams = ast.literal_eval(f.read())
    print("--Raw streaming data loaded!")

    # subset unique tracks and get features
    artist_track = [{k: v for k, v in d.items() if (k != 'endTime') & (k != 'msPlayed')} for d in streams]
    unique_tracks = [dict(y) for y in set(tuple(x.items()) for x in artist_track)]
    print("Getting features from Spotify API for {} unique tracks...".format(len(unique_tracks)))
    for song in unique_tracks:
        id, date = sm.get_api_id_date(track_name = song['trackName'], 
                                      artist = song['artistName'],
                                      token = token)
        if isinstance(id, str):
            features, genres = sm.get_api_features(id, token)
            if isinstance(features, dict):
                features.update({'release_date': date, 'genres': genres})
                song.update(features)
    print("--Track features recieved!")

    # export features as csv
    print("Exporting streaming and feature data as .csv files...")
    df = pd.DataFrame(unique_tracks)
    df.to_csv('data/processed/track_features.csv', 
              index = False)

    # export streaming history as csv
    df_history = pd.DataFrame(streams)
    df_history.to_csv('data/processed/streaming_history.csv',
                      index = False)
    print("--Export complete!")

    # generate outcomes
    print("Generating and exporting outcomes...")

    # import raw playlist data
    with open('data/raw/Playlist1.json', 'r') as f:
        playlist_raw = json.load(f)

    # obtain playlist outcome
    playlist_tracks = sm.getPlaylistOutcome(playlist_raw, good_playlist_names)

    # organize training data
    X, X_columns, y_score, y_playlist = sm.getTrainingData(df_history, df, playlist_tracks)

    # save training data as csv
    X_df = pd.DataFrame(X, columns = X_columns)
    y_score = pd.DataFrame(y_score, columns = ['y_score'])
    y_playlist = pd.DataFrame(y_playlist, columns = ['y_playlist'])

    X_df.to_csv('data/processed/X.csv', index=False)
    y_score.to_csv('data/processed/y_score.csv', index=False)
    y_playlist.to_csv('data/processed/y_playlist.csv', index=False)
    print("--Outcomes generated and exported!")

if __name__ == '__main__':
    main()
