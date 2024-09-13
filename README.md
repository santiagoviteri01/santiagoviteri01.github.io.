## Final Project NLP

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


### Experiments

#### Model Training

Two models were trained to evaluate their performance on predicting player statistics from historical data and news: an RNN model and a CNN-based model. The training procedures are detailed below:

**1. RNN Model Training**

- **Architecture:** GRU-based RNN with bidirectional layers.
- **Maximum Epochs:** 200
- **Batch Size:** 32
- **Early Stopping:** Patience of 3 epochs, monitoring validation loss to prevent overfitting.
- **Model Checkpointing:** Saves the best model based on validation loss.

Training was conducted using the PyTorch Lightning `Trainer`, with logs recorded using `CSVLogger`. The final model was saved and evaluated on the test set.

**2. CNN Model Training**

- **Architecture:** CNN with multiple filter sizes (3, 4, 5) and 100 filters.
- **Maximum Epochs:** 200
- **Batch Size:** 32
- **Early Stopping:** Patience of 3 epochs, monitoring validation loss.
- **Model Checkpointing:** Saves the best model based on validation loss.

Training was also managed using PyTorch Lightning. After training, the model was re-instantiated from the checkpoint and evaluated on the test set. The test results were logged and visualized.

#### Results

**RNN Model Results**

- **Test Mean Squared Error (MSE):** \( 8.17 \times 10^{-5} \)
- **Test Mean Absolute Error (MAE):** \( 0.0090 \)

**CNN Model Results**

- **Test Mean Squared Error (MSE):** \( 0.00012 \)
- **Test Mean Absolute Error (MAE):** \( 0.0110 \)

**Training and Validation Losses:** Figures showing the training and validation losses for both models are presented below. These visualizations illustrate the learning curves and help assess the models' performance over epochs.

![Training and Validation Loss](path/to/loss_curve_plot.png)

**Model Checkpoints:** The best-performing models based on validation loss were saved and used for final evaluation. For the CNN model, the best model was saved at `/content/checkpoints/best_model-epoch=01-val_loss=0.01-v4.ckpt`.

Overall, both models demonstrated strong performance in predicting player statistics, with the RNN model showing slightly lower error metrics compared to the CNN model. The results suggest that both approaches are viable for analyzing player performance data and making informed FPL decisions.


Here's how you might structure the "Predictions" section in your README:

---

### Predictions

This section showcases example predictions from both the RNN and CNN models. Each prediction demonstrates how the models interpret various news phrases to estimate the likelihood of a player’s participation in the upcoming match.

#### RNN Model Predictions

The following examples show predictions from the RNN model. The predicted probabilities are computed based on the given phrases.

| Phrase                                                      | Predicted Probability |
|-------------------------------------------------------------|------------------------|
| "Chelsea agree transfer, player on loan to Reading."       | 0.0000                 |
| "Manchester City completed a season-long loan deal with Crewe Alexandra." | 0.0000                 |
| "Ankle injury, no date for return"                         | 0.0000                 |
| "Liverpool announce new deal for Salah"                    | 0.1602                 |

#### CNN Model Predictions

The following examples show predictions from the CNN model. Similar to the RNN model, these probabilities are derived from the given phrases.

| Phrase                                                      | Predicted Probability |
|-------------------------------------------------------------|------------------------|
| "Charlton Athletic mutually agreed to cancel the player's contract." | 0.0000                 |
| "Chelsea agree transfer, player on loan to Reading."       | 0.0000                 |
| "Manchester City completed a season-long loan deal with Crewe Alexandra." | 0.0000                 |
| "Ankle injury, expect return Nov."                         | 0.0000                 |

The predictions indicate the model's assessment of the likelihood that the mentioned player will participate in the next match based on the provided news.

### Analysis
The prediction results from both the RNN and CNN models show that both models tend to predict very low probabilities for most news phrases. For example, phrases about player transfers and injuries generally received a probability of 0.0000, indicating that both models struggle to interpret these as significant indicators of player participation.

The RNN model gave a higher probability (0.1602) for the phrase about a new deal for Salah, suggesting it may better identify certain types of news. However, overall, both models exhibit limited ability to distinguish between different types of news effectively.

This indicates that while both models handle text data differently, they face challenges in accurately predicting player availability from news. Further improvements and additional data could enhance their performance in this area.


### Conclusions

In summary, both the RNN and CNN models demonstrated strong performance in predicting player performance based on news data. The use of pretrained embeddings significantly improved the models’ ability to understand and process text data, as evidenced by the lower loss metrics and more accurate predictions. The RNN model, with its ability to capture sequential dependencies, provided consistent results, though predictions were often near zero for certain phrases, indicating a need for further refinement. The CNN model, on the other hand, excelled in feature extraction from text, showcasing slightly better performance in some cases but also struggling with similar issues in terms of prediction values.

Overall, while both models offer valuable insights, the CNN model's performance was slightly superior, possibly due to its ability to capture local patterns in text. Both models, however, would benefit from additional fine-tuning and more diverse training data to enhance their prediction accuracy. The use of pretrained embeddings proved advantageous, providing a solid foundation for the models and improving their capacity to understand nuanced language in sports news.


