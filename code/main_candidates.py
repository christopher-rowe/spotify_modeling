#!/usr/bin/env python3
# File name: main_candidates.py
# Description: Identify and add candidates to data_candidates spotify playlist
# Author: Chris Rowe
# Date: 08-08-2020

import os
import pandas as pd
import numpy as np
import pickle
import get_features as gf
import find_candidates as fc
import generate_training_data as gtd
from config import *


def main():

    # import models
    print("Importing models...")
    os.chdir(os.path.dirname(os.getcwd()))
    xgb_stage1_playlist_model = pickle.load(open('saved_models/xgb_playlist_model_wg.sav', 'rb'))
    xgb_stage1_score_model = pickle.load(open('saved_models/xgb_score_model_wg.sav', 'rb'))
    print("--Models imported!")

    # get spotify authentication token
    print("Getting the spotify authentication token...")
    auth, token, refresh_token = gf.get_token(username, client_id, 
                                              client_secret, redirect_uri, scope)
    print("--Token recieved!")
   
    # Processing data playlists and fitting stage 2 model
    print("Processing data playlists and fitting stage 2 model...")
    X_data_playlists, y_data_playlists = fc.getDataPlaylistXY(auth, token, refresh_token)
    xgb_stage2_model = fc.fitDataPlaylistModel(X_data_playlists, y_data_playlists)
    print("--Stage 2 model ready!")

    # obtain stage 1 and stage 2 training features
    stage1_features = list(pd.read_csv('training_features/stage1_training_features.csv', names=['x']).x)
    stage2_features = list(pd.read_csv('training_features/stage2_training_features.csv', names=['x']).x)

    # identify candidates and push to spotify playlist
    print("Obtaining Random Tracks, fitting models, and retaining top candidates...")
    n_iter = 10
    for __ in range(n_iter):
        print('Iteration: ' + str(__) + ' of ' + str(n_iter))
        all_random_tracks = []
        all_random_track_genres = []
        while len(all_random_tracks) < 500:
            id, uri, name, artist = fc.getRandomTrack(auth, token, refresh_token)
            features, genres = gf.get_api_features(id, auth, token, refresh_token)
            if isinstance(features, dict):
                new_record = [id, uri, name, artist] + list(features.values())[0:11]
                all_random_tracks.append(new_record)
                all_random_track_genres.append(genres)
        columns = ['id', 'uri', 'track', 'artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        all_random_tracks = pd.DataFrame(all_random_tracks, 
                                        columns = columns) 
        X_random = all_random_tracks[columns[4:15]]

        # reconcile genres so they match those used in training model
        genre_dummies = gtd.getGenreDummies(all_random_track_genres)
        genre_dummies_stage1 = fc.reconcileGenres(genre_dummies, stage1_features[11:])
        genre_dummies_stage2 = fc.reconcileGenres(genre_dummies, stage2_features[11:])

        # generate stage1 and stage2 X matrices with appropriate genres and feature order
        X_random_stage1 = pd.concat((X_random, genre_dummies_stage1), axis = 1)
        X_random_stage1 = X_random_stage1[stage1_features]
        X_random_stage2 = pd.concat((X_random, genre_dummies_stage2), axis = 1)
        X_random_stage2 = X_random_stage2[stage2_features]

        # predict stage 1 outcomes
        stage1_playlist_p = np.array([item[1] for item in xgb_stage1_playlist_model.predict_proba(X_random_stage1)]) 
        stage1_score_p = xgb_stage1_score_model.predict(X_random_stage1)

        # predict stage 2 outcomes
        stage2_p = np.array([item[1] for item in xgb_stage2_model.predict_proba(X_random_stage2)])
        all_random_tracks['total_playlist_p'] = stage1_playlist_p*stage2_p
        all_random_tracks['total_score_p'] = stage1_score_p*stage2_p
        candidates_playlist = list(all_random_tracks.sort_values('total_playlist_p', ascending=False).iloc[0:5, 1])
        candidates_score = list(all_random_tracks.sort_values('total_score_p', ascending=False).iloc[0:5, 1])
        candidates = list(set(candidates_playlist + candidates_score))
        fc.addCandidates(auth, token, refresh_token, candidates, target_playlist)

    print("Candidate search complete, playlist updated!")

if __name__ == '__main__':
    main()