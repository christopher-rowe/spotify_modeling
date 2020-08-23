
# File name: config.py
# Description: To initialize spotify authenticaion parameters
#              and other objects
# Author: Chris Rowe

# initialize authentication parameters
username = 'xxx' # Spotify username
client_id ='xxx' # Spotify cliend id from registered app
client_secret = 'xxx' # Spotify cliend secret from registered app
redirect_uri = 'xxx' # arbitrary redirect_uri from registered app 
scope = 'xxx' # scope (see API documentation for details)

# initialize good playlist names
# description: this is a list of personal playlists that are used
#              to engineer the stage 1 playlist outcome
good_playlist_names = ['playlist1', 'playlist2', 'playlist3']

# target playlist id for data candidates (obtainable via spotify API)
target_playlist = 'xxx'
