
"""GraphNLI Inference Pipeline"""

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
gamma = 0.2

model_save_path = 'output/www_banepo_st_' + model_name.replace("/", "-")

train_samples = []
test_samples = []

testset = pd.read_csv('./pleroma_toxic_random_walks.csv')
testset = testset.fillna('')

print(testset.shape)

for i in range(len(testset)):
    texts = []
    for j in range(1, 5):
        texts.append(testset.iloc[i]['sent' + str(j)])
    test_samples.append(InputExample(texts=texts, label=float(testset.iloc[i]['toxicity'])))

test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=train_batch_size, drop_last=False)

# Load the model and evaluate its performance on the test dataset.

model = SentenceTransformer(model_save_path)
model.to(torch.device('cpu'))

test_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2, gamma=gamma)
test_evaluator = LabelAccuracyEvaluator(test_dataloader, name='sts-test', softmax_model=test_loss)
test_evaluator(model, output_path=model_save_path)
