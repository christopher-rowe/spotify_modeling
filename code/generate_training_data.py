import os
import pandas as pd
import numpy as np
import json

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
    
    # check if input object is series and conver to list
    if isinstance(genres, pd.Series):
        genres = list(genres)

    # archive code (when series was brought from csv) 
    #all_genres = [eval(item) if type(item) == str else item for item in all_genres]
    #all_genres = [['none'] if type(item) == float else item for item in all_genres]

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

    # identify genres with < 10 songs
    rare_genres = list(genre_dummies.sum(axis=0).loc[genre_dummies.sum(axis=0) < 10].index)

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
