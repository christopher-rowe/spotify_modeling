# Spotify Modeling
Using the Spotify API and machine learning to find new music

NOTE: I'm frequently tinkering with this repository to try and improve performance

## Overview
Spotify has a lot of cool features that casual users might not be aware of. First, any Spotify user can download their streaming history (for the past 1-2 years), their playlists, and some other stuff related to their account. Second, the company has some pretty incredible API's for developers and other curious folks. I don't have a lot of experience working with API's but I personally found Spotify's to be very intuitive and accessible. Third, Spotify has come up with some interesting ways to characterize tracks and artists. For tracks, there are about a dozen audio features that characterize the sound of a song (e.g., acousticness, danceability, energy, instrumentalness, loudness). 

I thought it would be fun to leverage my streaming history, Spotify's API's, and these audio features to try and construct models that might be able to predict how much I will like a song, and then use these models to identify new music to listen to. 

## Current Modeling Strategy
Details TBD

## Organization
- /data: 
- /code: 

## Qualitative Progress Notes:
- First iteration: I used only the models fit on my streaming history to identify candidate tracks, I didn't incorporate genre into the models, and I retained the top 5\% performing tracks as candidates. I got some good candidates, but I got more bad candidates, and I got a really high proportion of classical songs (e.g., Bach, Mozart).
- Second iteration: I incorporated genre into the streaming history models (thinking this might suppress the classical songs), incorporated the stage 2 model that uses the first round of candidates as training data, and I retained the top 1\% of performing tracks (thinking if I'm more exclusive, I could weed out some more questionable songs). The percentage of good candidates seemed to improve, no more classical music, but a disproportionately large (relative to what I would like to see) number of rap songs. I also noticed that I'm getting songs that I'm sure I've listened to in the last few years (which defeats the purpose of finding "new" music), so I intend on ultimately stripping those tracks as candidates, but figured I'd let the model choose these for now.