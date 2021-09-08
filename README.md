# lsml2-final-project-ilya-mansurov

## Dataset

Dataset contains information about user ratings (like = 1 / dislike=-1) of movies and series at smotrim.ru.
Dataset have been cleared of empty data, repeating votes, users and items with less than 3 votes.
Вataset consists of 44387 ratings, 1995 items and 10,076 users. Classes can be considered as balanced

[Dataset](./data/raw_ratings.csv)

## Task

My objective is to develop recommendation service. Service will recommend top K items for particular user with highest predicted rating.

## Model

We will solve rating prediction as a binary classification problem. We will use MLP as a [model] (./MLP.py).
The model receives user and item identifiers as input, then embeddings are built for them.
The vector from embeddings goes through 2 fully connected layers with RELU and dropout and sigmoid on the last layer.
Thus, at the output, the model gives the probabilities of a positive class. BCEloss is used as the loss function,
and roc_auc_score is used as the metric.

## Trainig

Model was trained with pytorch_lightning framework, the results of the experiments were stored with MLFlow.
The experiments were carried out with several options for hyperparameters - the number of fully connected layers,
it's size, lerning rate.

## Service architecture

Final model wrapped in a web service with MLFlow. Main service process requests in asynchronous manner with celery
framework with redis as backend storage and message broker.