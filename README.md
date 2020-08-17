# Spotify Modeling
Using the Spotify API and machine learning to find new music

**NOTE:** I'm frequently tinkering with this repository to try and improve performance

## Overview
Spotify has a lot of cool features that casual users might not be aware of. First, any Spotify user can download their streaming history (for the past 1-2 years), their playlists, and some other stuff related to their account. Second, the company has some pretty incredible API's for developers and other curious folks. I don't have a lot of experience working with API's but I personally found Spotify's to be very intuitive and accessible. Third, Spotify has come up with some interesting ways to characterize tracks and artists. For tracks, there are about a dozen audio features that characterize the sound of a song (e.g., acousticness, danceability, energy, instrumentalness, loudness). In addition to the audio features, Spotify characterizes artists with over 1500 genres. These include broad categories such as "rock" or "hip-hop", but also include some rather niche categories such as "post-doom metal" and "new orleans rap".

I thought it would be fun to leverage my personal streaming history, Spotify's API's, and these audio features and genres to try and construct models that might be able to predict how much I will like a song, and then use these models to identify new music to listen to. 

## General Process
First iteration (populating initial candidate pool):
1. Obtain "random" tracks from entire Spotify library.
2. Use audio features and genres of the track to predict whether I will like them (details below)
3. Push tracks with highest predicted values to a personal Spotify playlist called "data_candidates"
4. Listen to the candidates, add them to a playlist called "data_1" if I like the track and one called "data_0" if I don't, and remove them from the "data_candidates" playlist.

Subsequent Iterations:
1. Obtain "random" tracks from entire Spotify library.
2. Use audio features and genres of the track to predict whether I will like them (details below)
3. Use audio features and genre of the tracks to predict whether the song would end up in "data_1" or "data_0"
4. Create a composite score using predicted values from the models in steps 2-3 above.
3. Push tracks with highest predicted values to a personal Spotify playlist called "data_candidates"
4. Listen to the candidates, add them to a playlist called "data_1" if I like the track and one called "data_0" if I don't, and remove them from the "data_candidates" playlist.
5. Repeat, tweaking various process details.

## Measures and Models
Details coming soon!

## Organization
Details coming soon!

## Qualitative Progress Notes:
- First cycle: I used only the models fit on my streaming history to identify candidate tracks, I didn't incorporate genre into the models, and I retained the top 5\% performing tracks as candidates. I got some good candidates, but I got more bad candidates, and I got a really high proportion of classical songs (e.g., Bach, Mozart).
- Second cycle: I incorporated genre into the streaming history models (thinking this might suppress the classical songs), incorporated the stage 2 model that uses the first round of candidates as training data, and I retained the top 1\% of performing tracks (thinking if I'm more exclusive, I could weed out some more questionable songs). The percentage of good candidates seemed to improve, no more classical music, but a disproportionately large (relative to what I would like to see) number of rap songs. I also noticed that I'm getting songs that I'm sure I've listened to in the last few years (which defeats the purpose of finding "new" music), so I intend on ultimately stripping those tracks as candidates, but figured I'd let the model choose these for now.