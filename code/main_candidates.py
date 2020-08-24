
# File name: main_candidates.py
# Description: Identify and add candidates to data_candidates spotify playlist
# Author: Chris Rowe

import os
import random
import pandas as pd
import numpy as np
import pickle
import spotify_modeling as sm
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
    auth, token, refresh_token = sm.get_token(username, client_id, 
                                              client_secret, redirect_uri, scope)
    print("--Token recieved!")
   
    # Processing data playlists and fitting stage 2 model
    print("Processing data playlists and fitting stage 2 model...")
    X_data_playlists, y_data_playlists = sm.getDataPlaylistXY(auth, token, refresh_token)
    xgb_stage2_model = sm.fitDataPlaylistModel(X_data_playlists, y_data_playlists)
    print("--Stage 2 model ready!")

    # obtain stage 1 and stage 2 training features
    stage1_features_playlist = list(pd.read_csv('training_features/stage1_playlist_training_features.csv', names=['x']).x)
    stage1_features_score = list(pd.read_csv('training_features/stage1_score_training_features.csv', names=['x']).x)
    stage2_features = list(pd.read_csv('training_features/stage2_training_features.csv', names=['x']).x)

    # identify candidates and push to spotify playlist
    print("Obtaining Random Tracks, fitting models, and retaining top candidates...")
    n_iter = 5
    for __ in range(n_iter):
        print('Iteration: ' + str(__) + ' of ' + str(n_iter))
        all_random_tracks = []
        all_random_track_genres = []
        while len(all_random_tracks) < 500:
            id, uri, name, artist = sm.getRandomTrack(auth, token, refresh_token)
            features, genres = sm.get_api_features(id, auth, token, refresh_token)
            if isinstance(features, dict):
                new_record = [id, uri, name, artist] + list(features.values())[0:11]
                all_random_tracks.append(new_record)
                all_random_track_genres.append(genres)
        columns = ['id', 'uri', 'track', 'artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        all_random_tracks = pd.DataFrame(all_random_tracks, 
                                        columns = columns) 
        all_random_tracks.dropna(inplace=True)
        X_random = all_random_tracks[columns[4:15]]
        
        # initialize list of audio features to identify genre indices
        audio_features = ['danceability', 'energy', 'key',
                          'loudness', 'mode', 'speechiness',
                          'acousticness', 'instrumentalness',
                          'liveness', 'valence', 'tempo']

        # identify index of training features where genres begin (for reconciling genres between new tracks and training data)
        stage1_playlist_genre_i = [item in audio_features for item in stage1_features_playlist]
        stage1_playlist_genre_i = next(idx for idx, item in enumerate(stage1_playlist_genre_i) if item==False)
        stage1_score_genre_i = [item in audio_features for item in stage1_features_score]
        stage1_score_genre_i = next(idx for idx, item in enumerate(stage1_score_genre_i) if item==False)
        stage2_genre_i = [item in audio_features for item in stage2_features]
        stage2_genre_i = next(idx for idx, item in enumerate(stage2_genre_i) if item==False)

        # reconcile genres so they match those used in training model
        genre_dummies = sm.getGenreDummies(all_random_track_genres)
        genre_dummies_stage1_playlist = sm.reconcileGenres(genre_dummies, stage1_features_playlist[stage1_playlist_genre_i:])
        genre_dummies_stage1_score = sm.reconcileGenres(genre_dummies, stage1_features_score[stage1_score_genre_i:])
        genre_dummies_stage2 = sm.reconcileGenres(genre_dummies, stage2_features[stage2_genre_i:])

        # generate stage1 and stage2 X matrices with appropriate genres and feature order
        X_random_stage1_playlist = pd.concat((X_random, genre_dummies_stage1_playlist), axis = 1)
        X_random_stage1_playlist = X_random_stage1_playlist[stage1_features_playlist]

        X_random_stage1_score = pd.concat((X_random, genre_dummies_stage1_score), axis = 1)
        X_random_stage1_score = X_random_stage1_score[stage1_features_score]

        X_random_stage2 = pd.concat((X_random, genre_dummies_stage2), axis = 1)
        X_random_stage2 = X_random_stage2[stage2_features]

        # predict stage 1 outcomes
        stage1_playlist_p = np.array([item[1] for item in xgb_stage1_playlist_model.predict_proba(X_random_stage1_playlist)]) 
        stage1_score_p = xgb_stage1_score_model.predict(X_random_stage1_score)

        # predict stage 2 outcomes
        stage2_p = np.array([item[1] for item in xgb_stage2_model.predict_proba(X_random_stage2)])

        # calculate 2-stage score and playlist outcomes
        all_random_tracks['stage1_playlist_p'] = stage1_playlist_p
        all_random_tracks['stage1_score_p'] = stage1_score_p
        all_random_tracks['stage2_p'] = stage2_p
        all_random_tracks['total_score'] = stage1_score_p*stage2_p
        all_random_tracks['total_playlist'] = stage1_playlist_p*stage2_p

        # select top 5 candidates for combined stage1/stage2 scores and stage2 score only
        #candidates_stage1_playlist = list(all_random_tracks['uri'].loc[all_random_tracks['stage1_playlist_p']>0.5])
        candidates_total_score = list(all_random_tracks.sort_values('total_score', ascending=False).iloc[0:5, 1])
        candidates_total_playlist = list(all_random_tracks.sort_values('total_playlist', ascending=False).iloc[0:5, 1])
        candidates_stage2 = list(all_random_tracks.sort_values('stage2_p', ascending=False).iloc[0:5, 1])
        candidates = list(set(candidates_total_score + candidates_total_playlist + candidates_stage2))
        sm.addCandidates(auth, token, refresh_token, candidates, target_playlist)

    print("Candidate search complete, playlist updated!")

if __name__ == '__main__':
    main()