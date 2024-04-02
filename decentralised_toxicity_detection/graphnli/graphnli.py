"""Latest code for GraphNLI model."""

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from LabelAccuracyEvaluator import *
from models import Dense
from SoftmaxLoss import *
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import math
import sys
import os
import gzip
import csv


model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'

train_batch_size = 8


model_save_path = 'output/freethinkers_lgbt_fed_' + model_name.replace("/", "-")


word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

dense_model = Dense.Dense(in_features=3*768, out_features=2)


train_samples = []
dev_samples = []
test_samples = []
trainset = pd.read_csv('../train_graph_set_walk.csv')
trainset = trainset.fillna('')


print(trainset.shape)

# Shuffle the dataset
trainset = trainset.sample(frac=1)

for i in range(len(trainset)):
    texts = []
    for j in range(1, 5):  # 6 for graph walk and 5 for random walk.
        texts.append(trainset.iloc[i]['sent' + str(j)])
    train_samples.append(InputExample(texts=texts, label=float(trainset.iloc[i]['toxicity'])))

test_samples = train_samples[ : math.ceil(0.2 * len(train_samples)) ]
train_samples = train_samples[math.ceil(0.2 * len(train_samples)) : ]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

for i in range(1):
    gamma = 0.2

    model = SentenceTransformer('output/shitposter_club_distilroberta-base/')
    train_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2, gamma=gamma)

    dev_evaluator = LabelAccuracyEvaluator(test_dataloader, name='sts-dev', softmax_model=train_loss)

    # Configure the training
    num_epochs = 3

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )


    # test_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, name='sts-test', softmax_model=train_loss)
    test_evaluator(model, output_path=model_save_path)
