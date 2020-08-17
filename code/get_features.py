#!/usr/bin/env python3
# File name: get_features.py
# Description: To initialize spotify authenticaion parameters
# Author: Chris Rowe
# Date: 04-07-2020

# import libraries
import json
from spotipy.oauth2 import SpotifyOAuth
import os
import ast
import pandas as pd
import requests
import spotipy
import urllib

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

def getPlaylistNames():

    # import playlist data
    os.chdir(os.path.dirname(os.getcwd()))
    with open('data/raw/Playlist1.json', 'r') as f:
        playlist_raw = json.load(f)

    # extract playlist names
    playlist_names = []
    for i in range(len(playlist_raw['playlists'])):
        playlist_names.append(playlist_raw['playlists'][i]['name'])

    # return playlist names
    return playlist_names

def getPlaylistOutcome(good_playlist_names):

    # import playlist data
    os.chdir(os.path.dirname(os.getcwd()))
    with open('data/raw/Playlist1.json', 'r') as f:
        playlist_raw = json.load(f)

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

def getTrainingData():

    # load streaming history and track features from csv
    os.chdir(os.path.dirname(os.getcwd()))
    df_history = pd.read_csv('data/processed/streaming_history.csv')
    df = pd.read_csv('data/processed/track_features.csv')

    # generate release year column
    df['release_year'] = df['release_date'].astype(str).str[0:4].astype(float)

    # keep minimal columns
    df_columns = ['artistName', 'trackName', 'id',
                'danceability', 'energy', 'key', 
                'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo',
                'duration_ms', 'release_year']
    df = df[df_columns]

    df_history_columns = ['artistName', 'trackName',
                        'endTime', 'msPlayed']
    df_history = df_history[df_history_columns]

    # subset tracks with features
    tracks_w_features = df.loc[df['id'].isna() == False, :].copy()

    # calculate likeability score
    min_features = tracks_w_features[['artistName', 'trackName', 'id', 'duration_ms']]
    outcomes = pd.merge(df_history, min_features, on = ['artistName', 'trackName'])
    outcomes['p_played'] = outcomes['msPlayed'] / outcomes['duration_ms']
    outcomes = outcomes[['id', 'artistName', 'trackName', 'p_played']]
    outcomes = outcomes.groupby(['id', 'artistName', 'trackName'], as_index = False).sum()
    outcomes = outcomes.rename(columns={'id': 'id', 
                                        'artistName': 'artistName',
                                        'trackName': 'trackName',
                                        'p_played': 'score'})
    outcomes['score'] = (outcomes['score'] - outcomes['score'].min()) / outcomes['score'].max()

    # obtain playlist outcome
    playlist_tracks = getPlaylistOutcome(good_playlist_names)

    # merge likeability score with playlist outcome
    outcomes = pd.merge(outcomes, playlist_tracks, how = 'left',
                        on = ['artistName', 'trackName'])

    # impute zero values for playlist outcome
    outcomes.loc[outcomes['playlist'].isna(), 'playlist'] = 0

    # merge features and outcomes
    final = pd.merge(tracks_w_features, outcomes, on = ['id'])

    # subset X and y's
    X = final[['danceability', 'energy', 'key', 
            'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo']].to_numpy()
    y_score = final['score'].to_numpy()
    y_playlist = final['playlist'].to_numpy()

    # return objects
    return X, y_score, y_playlist