# File name: config.py
# Description: To initialize spotify authenticaion parameters
# Author: Chris Rowe

# initialize authentication parameters
username = 'crowe@ucla.edu'
client_id ='fcac9ea643224c98a34ab59b6b826178'
client_secret = '4026fe3681a64c228597fd34ac7b4671'
redirect_uri = 'http://localhost:7777/callback'
scope = 'user-read-recently-played playlist-read-private playlist-modify-private'

# initialize good playlist names
# description: this is a list of personal playlists that are used
#              to engineer the stage 1 playlist outcome
good_playlist_names = ['KM', 'if then was now', '2020',
                       'Bad Ass Folk Women', 'Robotix',
                       'BM', 'psychedelic', 'cloud',
                       'BRC 2019', '1964', '1963-1969',
                       'HEADY SHIT', '2019', '2018',
                       '2017', '2016', 'August 2015',
                       'B', 'A']

# this is the playlist id for the data_candidates playlist
target_playlist = '69SFgw78N4seDVXqZNTPa8'

# archive code to get list of user's playlists (this is helpful for identifying the 
#   name of the data_candidates playlist)
#url = 'https://api.spotify.com/v1/users/' + userid + '/playlists'
#headers = {
#'Accept': 'application/json',
#'Content-Type': 'application/json',
#'Authorization': f'Bearer ' + token,
#}
#response = requests.get(url, headers = headers, timeout = 5)  
#response.json()['items'][0]