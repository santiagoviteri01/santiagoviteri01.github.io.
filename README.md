## Introduction
This project focuses on predicting the likelihood of player availability for football matches based on text news reports. We use Natural Language Processing (NLP) to analyze the news headlines and apply both RNN (LSTM) and CNN models to obtain the probability of the player being available. Both models are designed to extract relevant features from text and use them to predict player availability.

## Problem Statement
In Fantasy Premier League (FPL), making informed decisions is key to success. While player statistics are valuable, real-time news on injuries, transfers, and team updates can significantly impact player performance. This project aims to analyze relevant news to provide FPL managers with actionable insights, helping them make better decisions on transfers, captain choices, and squad selection.

## Dataset 

The dataset used in this study comprises historical player statistics from the 2016-2017 to 2023-2024 seasons sourced from Vaastav’s GitHub repository [site](https://github.com/vaastav/Fantasy-Premier-League). Features include player
attributes such as goals scored, assists, minutes played, clean
sheets, and other relevant metrics recorded per match. How-
ever, the main variables to consider are the news and chance_of_playing_this_round since I aim to find how a news could give insights of the player starting in the following match.
