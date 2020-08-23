# Spotify Modeling
Using personal streaming history, the Spotify API, and machine learning to find new music.

A couple of important points: 
1. I am not a software engineer, so this repository is best for folks who are interested in engaging and experimenting with the code and process, as opposed to those looking for a nicely packaged product.
2. I'm frequently tinkering with this repository to try and make improve performance, so it's definitely not static.

## Overview
Spotify has a lot of cool features that casual users might not be aware of. First, any Spotify user can download their streaming history (for the past 1-2 years), their playlists, and some other stuff related to their account. Second, the company has some pretty incredible API's for developers and other curious folks. I don't have a lot of experience working with API's but I personally found Spotify's to be very intuitive and accessible. Third, Spotify has come up with some interesting ways to characterize tracks and artists. For tracks, there are about a dozen audio features that characterize the sound of a song (e.g., acousticness, danceability, energy, instrumentalness, loudness). In addition to the audio features, Spotify characterizes artists with over 1500 genres. These include broad categories such as "rock" or "hip-hop", but also include some pretty niche categories such as "post-doom metal" and "new orleans rap".

I thought it would be fun to leverage my personal streaming history, Spotify's API's, and these audio features and genres to try and construct models that might be able to predict how much I will like a song, and then use these models to identify new music to listen to. 

## General Process
Broadly, I've written some code to grab "random" songs from the vast Spotify library and use their audio characteristics and genres to predict how much I will like them. High performers get pushed to a "candidates" playlist, which I listen to at my leisure in the Spotify app and sort into one of two playlists to indicate whether or not I like the song. Subsequent iterations incorporate this newly labeled data in the the process, with the hope that the more candidate that I listen to and label, the better the model will be able to match my tastes. The jury is still out as to how well this works, but it's fun to explore.

Regarding modeling, I've organized models into two stages:
- Stage 1: Includes two models that were fit only to my streaming history, not to candidates that are part of the active process. I engineered two outcomes  from the streaming history: 1) whether or not I had added the track to a playlist and 2) the proportion of the track that I listened to, summed up over each time I listened to the track. Model predictions for these outcomes correspond to a "playlist" score and a "likeability" score. Neither of these are perfect indicators of how much I like a song, but they seemed like reasonable places to start.
- Stage 2: Includes one model that is fit to the candidate tracks that I have listened to and categorized as either "like" or "dislike" by manually sorting them into one of two playlists.

I have not yet identified the best approach, but the songs that make it into the candidates playlist have high predicted values from some sort of combination of these three models.

Here is a more detailed outline of the process:

**Prior to first iteration:**
1. Obtain audio features and genres for all tracks in streaming history. 
2. Engineer playlist and likeability score outcomes from streaming history.
3. Fit stage 1 models to streaming history data.

**First iteration (populating initial candidate pool):**
1. Obtain "random" tracks from entire Spotify library.
2. Use stage 1 models to predict playlist and likeability score outcomes.
3. Push tracks with highest predicted values to a personal Spotify playlist called "data_candidates".
4. Listen to the candidates, add them to a playlist called "data_1" if I like the track and one called "data_0" if I don't, and remove them from the "data_candidates" playlist.

**Subsequent Iterations:**
1. Obtain tracks and their features/genres from the "data_1" and "data_0" playlists.
2. Fit stage 2 model to predict the probability that a track made it into the "data_1" playlist. 
3. Obtain "random" tracks from entire Spotify library.
4. Use both stage1 and stage 2 models to predict relevant outcomes.
5. Create one or more composite scores using predicted values from the stage 1 and stage 2 models.
3. Push tracks with highest predicted values to a personal Spotify playlist called "data_candidates"
4. Listen to the candidates, add them to a playlist called "data_1" if I like the track and one called "data_0" if I don't, and remove them from the "data_candidates" playlist.
5. Repeat, tweaking various process details.

## Organization
- \code
	- spotify_modeling.py : includes all custom functions for the project
	- main_features.py : script for getting audio features and genres for all tracks in streaming history
	- main_stage1_training_data.py : script for processing streaming history into objects for stage 1 model training
	- main_stage1_models.py : script for stage 1 feature selection and model fitting
	- main_candidates.py : script for identifying new candidates and adding them to "data_candidates" Spotify playlist.
- \saved_models : directory for saved stage 1 model objects
- \training_features: directory for vectors of features included in each of the three models 
- \data
	- \processed:
		- streaming_history.csv : my personal streaming history as processed in main_features.py
		- track_features.csv : unique tracks from my streaming history with audio features and genres as processed in main_features.py
		- X.csv : full stage 1 design matrix for stage 1 models as processed in main_stage1_training_data.py
		- y_playlist.csv : stage 1 playlist outcome as processed in main_stage1_training_data.py
		- y_score.csv : stage 1 likeability score outcome as processed in main_stage1_training_data.py

## Qualitative Progress Notes:
- First cycle: I used only the models fit on my streaming history to identify candidate tracks, I didn't incorporate genre into the models, and I retained the top 5\% performing tracks as candidates. I got some good candidates, but I got more bad candidates, and I got a really high proportion of classical songs (e.g., Bach, Mozart).
- Second cycle: I incorporated genre into the streaming history models (thinking this might suppress the classical songs), incorporated the stage 2 model that uses the first round of candidates as training data, and I retained the top 1\% of performing tracks (thinking if I'm more exclusive, I could weed out some more questionable songs). The percentage of good candidates seemed to improve, no more classical music, but a disproportionately large (relative to what I would like to see) number of rap songs--like 90\%. I also noticed that I'm getting songs that I'm sure I've listened to in the last few years (which defeats the purpose of finding "new" music), so I intend on ultimately stripping those tracks as candidates, but figured I'd let the model choose these for now.