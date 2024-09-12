## Introduction
This project focuses on predicting the likelihood of player availability for football matches based on text news reports. We use Natural Language Processing (NLP) to analyze the news headlines and apply both RNN (LSTM) and CNN models to obtain the probability of the player being available. Both models are designed to extract relevant features from text and use them to predict player availability.

The dataset used in this study comprises historical player statistics from the 2016-2017 to 2023-2024 seasons sourced from Vaastav’s GitHub repository [site](https://github.com/vaastav/Fantasy-Premier-League). Features include player
attributes such as goals scored, assists, minutes played, clean
sheets, and other relevant metrics recorded per match. How-
ever, the main variables to consider are the news and chance_of_playing_this_round since I aim to find how a news could give insights of the player starting in the following match.
