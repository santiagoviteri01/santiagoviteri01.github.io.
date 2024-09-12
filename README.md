## Introduction
This project focuses on predicting the likelihood of player availability for football matches based on text news reports. We use Natural Language Processing (NLP) to analyze the news headlines and apply both RNN (LSTM) and CNN models to obtain the probability of the player being available. Both models are designed to extract relevant features from text and use them to predict player availability.

## Problem Statement
In Fantasy Premier League (FPL), making informed decisions is key to success. While player statistics are valuable, real-time news on injuries, transfers, and team updates can significantly impact player performance. This project aims to analyze relevant news to provide FPL managers with actionable insights, helping them make better decisions on transfers, captain choices, and squad selection.



##  Dataset and Dataprocess

In this project, I used historical player statistics from the 2016-2017 to 2023-2024 seasons, which I obtained from Vaastav’s GitHub repository ([Vaastav's FPL Data Repository](https://github.com/vaastav/Fantasy-Premier-League)). The dataset contains various player attributes, such as goals scored, assists, and clean sheets. My main focus is on analyzing how news articles and the `chance_of_playing_this_round` variable can offer insights into a player's likelihood of starting in an upcoming match.

Here’s how I processed the textual data:

#### Tokenization and Vocabulary Building

1. **Preprocessing and Tokenization:**
   - I tokenized the news articles using the Natural Language Toolkit (NLTK). The tokenization process involved splitting each news article into words and filtering out numeric tokens (like scores or percentages). I also applied stemming to reduce words to their base forms, which helps in standardizing the text data.

2. **Creating the Vocabulary:**
   - I built a vocabulary from the training data, mapping each unique word to an integer index. This step is essential for converting the text into a format suitable for deep learning models. The vocabulary excludes numeric tokens and includes a special `<unk>` token to handle any words not seen during training.

3. **Embedding Initialization:**
   - To initialize the embedding layer of my models, I used pre-trained word embeddings. These embeddings provide dense vector representations of words, capturing their semantic meaning. I loaded these embeddings from a file and mapped each word in the vocabulary to its corresponding vector.

#### Data Preparation

After tokenizing the text data and converting it into sequences of integer indices, I padded these sequences to ensure they all have the same length. This padding helps in maintaining consistency when training the models.

I then split the dataset into training, validation, and test sets using an 80/10/10 split. For each set, I created a dataset object that handles data loading and batching during model training and evaluation.

By transforming the news data into tokenized sequences and using embeddings, I aim to extract useful information that can indicate whether players are likely to start in upcoming matches. This approach is designed to help FPL managers make more informed decisions.

## Model Architectures
### RNN Model
The RNN model is designed to capture the temporal dependencies in text data from news articles that impact FPL player performance and decision-making. It helps understand how sequential context in the text (like match updates or injury reports) can influence FPL strategies.

-Input Layer: Tokenized and preprocessed text sequences representing news articles or reports are input into the model.

-Embedding Layer: Converts each word into dense vector representations of size 100 to capture semantic meaning.

-RNN Layers: The model includes two stacked RNN layers with 128 hidden units each, which processes the input sequentially. The recurrent layers capture temporal relationships between words in the news articles.

-Dropout Layer: Regularization is applied via a dropout layer with a dropout rate of 0.2 to prevent overfitting.

-Fully Connected (Dense) Layer: The output from the RNN layers is passed through a fully connected layer with a linear activation function.

### CNN Model
The CNN model is utilized to capture key phrases or patterns in news articles that are important for making FPL decisions. It is particularly useful for extracting non-sequential, spatial patterns in the text, such as keywords (e.g., “injury”, “suspension”) that may indicate significant changes in player status.

-Input Layer: Preprocessed news article text, converted into sequences of word embeddings, serves as the input.

-Convolutional Layers:

1. First Convolutional Layer: A 1D convolutional layer with 64 filters and a kernel size of 3 is applied to extract meaningful n-gram features from the text.

2. Second Convolutional Layer: Another 1D convolutional layer with 128 filters and the same kernel size refines the feature extraction, capturing more complex patterns in the text.

-Max-Pooling Layer: Following the convolutional layers, a max-pooling layer with a pool size of 2 reduces the dimensionality of the feature maps, summarizing the key information.

-Dropout Layer: A dropout rate of 0.3 is applied to reduce the risk of overfitting.

-Fully Connected (Dense) Layer: The output from the max-pooling layer is flattened and passed through a fully connected layer with 128 units.
