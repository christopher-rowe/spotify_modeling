import requests
import get_features as gf
import generate_training_data as gtd
from config import *
import string
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

def getRandomTrack(auth, token, refresh_token):
    
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    # initialize random search parameters
    random_query = random.choice(string.ascii_letters) + '%'
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
    
def addCandidates(token, candidates, playlist_id):

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
        token = gf.get_token(username, client_id, 
                             client_secret, redirect_url, scope)
        headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer ' + token,
        }
        requests.post(url, headers = headers, timeout = 5)    

def getDataPlaylistXY(token):

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
    features_0, genres_0 = zip(*[gf.get_api_features(x, token) for x in id_0])
    features_0 = pd.DataFrame(features_0)
    X_0 = features_0.iloc[:, 0:11]
    genre_dummies_0 = gtd.getGenreDummies(genres_0)    
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
    features_1, genres_1 = zip(*[gf.get_api_features(x, token) for x in id_1])
    features_1 = pd.DataFrame(features_1)
    X_1 = features_1.iloc[:, 0:11]
    genre_dummies_1 = gtd.getGenreDummies(genres_1)    
    X_1 = pd.concat((X_1, genre_dummies_1), axis = 1)

    # identify rare genres across both data_0 and data_1 tracks
    genre_dummies = pd.concat((genre_dummies_0, genre_dummies_1), axis = 0)
    genre_dummies.fillna(0, inplace=True)
    rare_genres = list(genre_dummies.sum(axis=0).loc[genre_dummies.sum(axis=0) < 5].index)

    # stack X_0 and X_1, drop rare genres, save features, convert to numpy array
    X = pd.concat((X_0, X_1), axis = 0)
    X.fillna(0, inplace=True)
    X.drop(rare_genres, axis=1, inplace=True)
    pd.DataFrame(X.columns).to_csv('training_features/stage2_training_features.csv',
                                               index=False, header=False)
    X = X.to_numpy()

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

    # identify optimal hyperparameters via random search
    xgb_param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5],
                    'n_estimators': [100, 500, 1000, 1500],
                    'subsample': [1.0, 0.75, 0.5],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [3, 5, 7, 9, 11],
                    'max_features': [None, 'sqrt']}

    # initialize model and grid or random search
    xgb_cls = GradientBoostingClassifier()
    xgb_cls_rs = RandomizedSearchCV(xgb_cls, xgb_param_grid, n_iter = 100, 
                                    n_jobs = -1, cv = 10)

    # Execute search
    xgb_cls_rs.fit(X, y)

    # save optimal model
    return xgb_cls_rs.best_estimator_
