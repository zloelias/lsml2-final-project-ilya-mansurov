# lsml2-final-project-ilya-mansurov

## Dataset

Dataset contains information about user ratings (like = 1 / dislike=-1) of movies and series on Smotrim.ru video platform.
Dataset have been cleared of empty ids, repeating votes, users and items with less than 3 votes.
Dataset consists of 44387 ratings, 1995 items and 10,076 users. Classes can be considered as balanced

[Dataset](./data/raw_ratings.csv)

## Task

My objective is to develop recommendation service. Service will recommend topK items for particular user with highest predicted rating.

## Model

We will solve rating prediction as a binary classification problem. We will use MLP as a [model] (./model.py).
The model receives user and item identifiers as input, then embeddings are built for them.
The vector from embeddings goes through 2 fully connected layers with RELU and dropout and sigmoid on the last layer.
Thus, at the output, the model gives the probabilities of a positive class. BCEloss is used as the loss function,
and roc_auc_score is used as the metric.

## Trainig

Model was [trained] (./train.ipynb) with pytorch_lightning framework, the results of the experiments were stored with MLFlow.
The experiments were carried out with several options for hyperparameters - the number of fully connected layers,
it's size, learning rate.

## Service architecture

Final model wrapped in a web service with [MLFlow] (./mlflow). Main [service] (./recommender) process requests in
asynchronous manner with celery framework with redis as backend storage and message broker.
To start service use `docker-compose up`. To set recomendation task `curl http://127.0.0.1:8081/count_recs_by_user/wKgcDlom3XR95yGlDpCGAg==
` or any other user_id from dataset, and then `curl http://127.0.0.1:8081/get_recs/<task_id>`
