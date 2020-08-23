
# File name: spotify_modeling.py
# Description: All custom functions for using streaming data to 
#              find new music, used in main_*.py files.
# Author: Chris Rowe

# import libraries
import json
from spotipy.oauth2 import SpotifyOAuth
import os
import ast
import pandas as pd
import numpy as np
import requests
import spotipy
import urllib
import random
import string
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from config import *

####################################################
# functions for accessing/navigating spotify api
####################################################

def get_token(username: str, 
              client_id: str,
              client_secret: str,
              redirect_uri: str,
              scope: str) -> str:
    """[summary]

    Args:
        user (str): your spotify user id 
        client_id (str): client_id for spotify web app
        client_secret (str): client secret for spotify web app
        redirect_url (str): redirect url for spotify web app
        scope (str): scope of authentication

    Returns:
        str: token needed for accessing spotify api
    """

    auth = SpotifyOAuth(client_id=client_id, 
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope=scope,
                        username=username)

    token = auth.get_access_token()['access_token']
    refresh_token = auth.get_access_token()['refresh_token']

    return auth, token, refresh_token

def get_api_id_date(track_name, artist, auth, token, refresh_token):
    """ Obtain track id and release date from spotify api

    Args:
        track_name (str): track name from streaming history
        token (str): spotify authentication token
        artist (str): track artist from streaming history

    Returns:
        str: returns spotify track id and release date, 
             or None and None if request is unsuccessful
    """
   
    # api request headers
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    
    # generate api request url
    url =  'https://api.spotify.com/v1/search?q=track:' + urllib.parse.quote(track_name, safe='') + '%20artist:' + urllib.parse.quote(artist, safe='') + '&type=track'

    # execute api request
    try:
        response = requests.get(url, headers = headers, timeout = 5)
        if response.status_code == 401:
            token = auth.refresh_access_token(refresh_token)['access_token']
            refresh_token = auth.refresh_access_token(refresh_token)['refresh_token']
            headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer ' + token,
            }
            response = requests.get(url, headers = headers, timeout = 5)
        json = response.json()
        first_result = json['tracks']['items'][0]
        track_id = first_result['id']
        release_date = first_result['album']['release_date']
        return track_id, release_date
    except:
        return None, None


def get_api_features(track_id, auth, token, refresh_token):
    """ obtain track features from spotify api

    Args:
        track_id (str): spotifyh track id
        token (str): spotify authentication token

    Returns:
        dict: spotify track features as dictionary   
    """

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    # get track features
    try:
        url =  'https://api.spotify.com/v1/audio-features/' + track_id
        response = requests.get(url, headers = headers, timeout = 5)  
        if response.status_code == 401:
            token = auth.refresh_access_token(refresh_token)['access_token']
            refresh_token = auth.refresh_access_token(refresh_token)['refresh_token']
            headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer ' + token,
            }
            response = requests.get(url, headers = headers, timeout = 5)
        features = response.json()
    except: 
        features = None

    # get artist genre
    try:
        url =  'https://api.spotify.com/v1/tracks/' + track_id
        response = requests.get(url, headers = headers, timeout = 5)  
        artist_id = response.json()['album']['artists'][0]['id']
        url =  'https://api.spotify.com/v1/artists/' + artist_id
        response = requests.get(url, headers = headers, timeout = 5)   
        genres = response.json()['genres']
    except: 
        genres = None

    return features, genres

####################################################
# functions for generating training data
####################################################

def getPlaylistNames(playlist_raw):
    
    # extract playlist names
    playlist_names = []
    for i in range(len(playlist_raw['playlists'])):
        playlist_names.append(playlist_raw['playlists'][i]['name'])

    # return playlist names
    return playlist_names

def getPlaylistOutcome(playlist_raw, good_playlist_names):

    # obtain list of playlist names
    playlist_names = []
    for i in range(len(playlist_raw['playlists'])):
        playlist_names.append(playlist_raw['playlists'][i]['name'])

    # identify indices of good playlists provided
    good_indices = []
    for name in good_playlist_names:
        good_indices.append(playlist_names.index(name))

    # initialize empty dataframe
    playlist_tracks = pd.DataFrame(columns=('trackName', 'artistName'))

    # populate dataframe with tracks from good playlists
    row = 0
    for i in good_indices:
        for j in range(len(playlist_raw['playlists'][i]['items'])):
            track = playlist_raw['playlists'][i]['items'][j]['track']['trackName']
            artist = playlist_raw['playlists'][i]['items'][j]['track']['artistName']
            playlist_tracks.loc[row] = [track, artist]
            row = row + 1

    # generate outcome as 1
    playlist_tracks['playlist'] = 1

    # drop duplicates
    playlist_tracks.drop_duplicates(inplace=True)

    # return
    return playlist_tracks

def getGenreDummies(genres):
    
    # check if input object is series and convert to list
    if isinstance(genres, pd.Series):
        genres = list(genres)
        genres = [eval(item) if type(item) == str else item for item in genres]
        genres = [['none'] if type(item) == float else item for item in genres]
    
    # add 'none' genre is there is no genre
    genres = [['none'] if (len(x) == 0 or x is None) else x for x in genres]

    # add underscores to single genres
    genres = [[x.replace(" ", "_") for x in z] for z in genres]

    # convert to dataframe and create dummies for each genre
    df = pd.DataFrame()
    df['genres'] = genres
    genre_dummies = pd.get_dummies(df['genres'].apply(pd.Series).stack()).sum(level=0)

    # return genre dummies as pandas dataframe
    return genre_dummies


def getTrainingData(streaming_history, track_features, playlist_tracks):

    # generate release year column
    track_features['release_year'] = track_features['release_date'].astype(str).str[0:4].astype(float)

    # keep minimal columns
    track_features_columns = ['artistName', 'trackName', 'id',
                'danceability', 'energy', 'key',
                'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo',
                'duration_ms', 'genres', 'release_year']
    track_features = track_features[track_features_columns]

    streaming_history_columns = ['artistName', 'trackName',
                        'endTime', 'msPlayed']
    streaming_history = streaming_history[streaming_history_columns]

    # subset tracks with features
    tracks_w_features = track_features.loc[track_features['id'].isna() == False, :].copy()

    # calculate likeability score
    min_features = tracks_w_features[['artistName', 'trackName', 'id', 'duration_ms']]
    outcomes = pd.merge(streaming_history, min_features, on = ['artistName', 'trackName'])
    outcomes['p_played'] = outcomes['msPlayed'] / outcomes['duration_ms']
    outcomes = outcomes[['id', 'artistName', 'trackName', 'p_played']]
    outcomes = outcomes.groupby(['id', 'artistName', 'trackName'], as_index = False).sum()
    outcomes = outcomes.rename(columns={'id': 'id',
                                        'artistName': 'artistName',
                                        'trackName': 'trackName',
                                        'p_played': 'score'})
    outcomes.score.replace(0, 
                       pd.Series(outcomes.score.unique()).nsmallest(2).iloc[-1],
                       inplace = True)                                    
    outcomes['score'] = np.log((outcomes['score']))
    outcomes['score'] = outcomes['score'] + np.abs(outcomes['score'].min())
    outcomes['score'] = outcomes['score']/outcomes['score'].max()

    # merge likeability score with playlist outcome
    outcomes = pd.merge(outcomes, playlist_tracks, how = 'left',
                        on = ['artistName', 'trackName'])

    # impute zero values for playlist outcome
    outcomes.loc[outcomes['playlist'].isna(), 'playlist'] = 0
    outcomes.drop(columns = ['artistName', 'trackName'], inplace = True)

    # merge features and outcomes
    final = pd.merge(tracks_w_features, outcomes, on = ['id'])

    # get genre dummies
    genre_dummies = getGenreDummies(final['genres'])

    # identify genres with < 20 songs
    rare_genres = list(genre_dummies.sum(axis=0).loc[genre_dummies.sum(axis=0) < 20].index)

    # drop rare genre dummies
    genre_dummies.drop(rare_genres, axis = 1, inplace=True)

    # subset X and y's
    X_columns = ['artistName', 'trackName', 'danceability',
                           'energy', 'key', 'loudness', 'mode',
                           'speechiness', 'acousticness',
                           'instrumentalness', 'liveness', 'valence', 'tempo']
    X = final[X_columns]
    X = pd.concat([X, genre_dummies], axis=1)
    X = X.to_numpy()
    y_score = final['score'].to_numpy()
    y_playlist = final['playlist'].to_numpy()

    # update columns
    X_columns = X_columns + list(genre_dummies.columns)

    # return objects
    return X, X_columns, y_score, y_playlist


####################################################
# functions for getting/adding candidates
####################################################

def getRandomTrack(auth, token, refresh_token):
    
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    # initialize random search parameters
    dice = random.choice(range(3))
    if dice == 0:
        random_query = random.choice(string.ascii_letters) + '%'
    elif dice == 1:
        random_query = '%' + random.choice(string.ascii_letters)
    else:
        random_query = '%' + random.choice(string.ascii_letters) + '%'
    random_offset = str(np.random.randint(1,2001))

    # generate url
    url =  'https://api.spotify.com/v1/search?q=track:' + random_query + '&type=track' + '&offset=' + random_offset + '&limit=50'

    # make request & obtain and return random track details
    try:
        response = requests.get(url, headers = headers, timeout = 5)
        if response.status_code == 401:
            token = auth.refresh_access_token(refresh_token)['access_token']
            refresh_token = auth.refresh_access_token(refresh_token)['refresh_token']
            headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer ' + token,
            }
            response = requests.get(url, headers = headers, timeout = 5)
        json = response.json()
        random_track = json['tracks']['items'][np.random.choice(len(json['tracks']['items']))]
        track_id = random_track['id']
        track_uri = random_track['uri']
        track_name = random_track['name']
        artist_name = random_track['artists'][0]['name']
        return track_id, track_uri, track_name, artist_name
    except:
        return None, None, None, None
    
def addCandidates(auth, token, refresh_token, candidates, playlist_id):

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    # concatenate candidates
    candidates = ','.join(candidates)
    
    # generate url
    url =  'https://api.spotify.com/v1/playlists/' + playlist_id + '/tracks?uris=' + candidates

    # add candidates to playlist
    try:
        requests.post(url, headers = headers, timeout = 5)    
    except:
        token = auth.refresh_access_token(refresh_token)['access_token']
        refresh_token = auth.refresh_access_token(refresh_token)['refresh_token']
        headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + token,
        }
        requests.post(url, headers = headers, timeout = 5)    

def getDataPlaylistXY(auth, token, refresh_token):

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    # process data_0 tracks and obtain X matrix
    url =  'https://api.spotify.com/v1/playlists/5Dm6quiW1b89HzOMgY083R/tracks'
    response = requests.get(url, headers = headers, timeout = 5)    
    id_0 = [x['track']['id'] for x in response.json()['items']]
    if response.json()['total'] > 100:
        for i in range(1, int(np.ceil(response.json()['total']/100))):
            offset = int(100*i)
            url =  'https://api.spotify.com/v1/playlists/5Dm6quiW1b89HzOMgY083R/tracks?offset=' + str(offset)
            response = requests.get(url, headers = headers, timeout = 5)    
            id_0.extend([x['track']['id'] for x in response.json()['items']]) 
    features_0, genres_0 = zip(*[get_api_features(x, auth, token, refresh_token) for x in id_0])
    features_0 = pd.DataFrame(features_0)
    X_0 = features_0.iloc[:, 0:11]
    genre_dummies_0 = getGenreDummies(genres_0)
    X_0 = pd.concat((X_0, genre_dummies_0), axis = 1)


    # process data_1 tracks and obtain X matrix
    url =  'https://api.spotify.com/v1/playlists/3qHfRMRSL8sVzV0z3devQf/tracks'
    response = requests.get(url, headers = headers, timeout = 5)    
    id_1 = [x['track']['id'] for x in response.json()['items']]
    if response.json()['total'] > 100:
        for i in range(1, int(np.ceil(response.json()['total']/100))):
            offset = int(100*i)
            url =  'https://api.spotify.com/v1/playlists/3qHfRMRSL8sVzV0z3devQf/tracks?offset=' + str(offset)
            response = requests.get(url, headers = headers, timeout = 5)    
            id_1.extend([x['track']['id'] for x in response.json()['items']])
    features_1, genres_1 = zip(*[get_api_features(x, auth, token, refresh_token) for x in id_1])
    features_1 = pd.DataFrame(features_1)
    X_1 = features_1.iloc[:, 0:11]
    genre_dummies_1 = getGenreDummies(genres_1) 
    X_1 = pd.concat((X_1, genre_dummies_1), axis = 1)

    # identify rare genres across both data_0 and data_1 tracks
    genre_dummies = pd.concat((genre_dummies_0, genre_dummies_1), axis = 0)
    genre_dummies.fillna(0, inplace=True)
    rare_genres = list(genre_dummies.sum(axis=0).loc[genre_dummies.sum(axis=0) < 20].index)

    # stack X_0 and X_1, drop rare genres, save features, convert to numpy array
    X = pd.concat((X_0, X_1), axis = 0)
    X.fillna(0, inplace=True)
    X.drop(rare_genres, axis=1, inplace=True)

    # generate y as numpy arrays
    y_0 = np.array([0] * X_0.shape[0])
    y_1 = np.array([1] * X_1.shape[0])
    y = np.concatenate((y_0, y_1))

    return X, y

def reconcileGenres(current_genre_dummies, target_genres):

        # identify intersecting genres
        genre_check_stage1a =  [item in target_genres for item in list(current_genre_dummies.columns)] 
        genre_check_stage1b =  [item in list(current_genre_dummies.columns) for item in target_genres] 

        # identify genres to drop from random tracks
        genres_r = pd.DataFrame({'genre': current_genre_dummies.columns, 'keep': genre_check_stage1a})
        genres_drop = list(genres_r.genre.loc[genres_r.keep == False])

        # identify genres to add to random tracks
        genres_mod = pd.DataFrame({'genre': target_genres, 'inv_add': genre_check_stage1b})
        genres_add = list(genres_mod.genre.loc[genres_mod.inv_add==False])
        
        # drop genres
        new_genre_dummies = current_genre_dummies.copy()
        new_genre_dummies.drop(genres_drop, axis=1, inplace=True)
        
        # add new genres (with values=0)
        for x in genres_add:
            new_genre_dummies[x] = [0] * new_genre_dummies.shape[0]

        # return current_genre_dummies
        return new_genre_dummies

def fitDataPlaylistModel(X, y):

    # feature selection:
    # initialize hyperparameter grids for random search
    xgb_param_grid={'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [100, 500, 1000],
                    'subsample': [1.0, 0.75, 0.5],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [3, 5, 7],
                    'max_features': [None, 'sqrt']} 

    # initialize model and random search
    xgb_cls_fs = RandomizedSearchCV(GradientBoostingClassifier(), 
                                    xgb_param_grid, n_iter = 50, 
                                    n_jobs = -1, cv = 10, 
                                    random_state = 6053)

    # execute search
    print("Fitting models for feature selection...")
    xgb_cls_fs.fit(X, y)

    # subset features based on feature importance
    feat_selector = SelectFromModel(xgb_cls_fs.best_estimator_, prefit=True)
    X_fs = feat_selector.transform(X)
    features = feat_selector.get_support()

    # main model
    # initialize model and random search
    xgb_cls = RandomizedSearchCV(GradientBoostingClassifier(), 
                                 xgb_param_grid, n_iter = 50, 
                                 n_jobs = -1, cv = 10, 
                                 random_state = 6053)

    # execute search
    print("Fitting final models...")
    xgb_cls.fit(X_fs, y)

    # save features 
    training_features = pd.DataFrame(list(X.columns[features]))
    training_features.to_csv('training_features/stage2_training_features.csv',
                             index=False, header=False)

    # save optimal model
    return xgb_cls.best_estimator_
