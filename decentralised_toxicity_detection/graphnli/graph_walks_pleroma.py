"""Using various Graph walks, generate Pleroma train and test sets."""

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import random
import math
from googleapiclient import discovery
from sentence_transformers import SentenceTransformer, util


api_key="<<ADD YOUR PERSPECTIVE API KEY>>"

client = discovery.build("commentanalyzer","v1alpha1",
                         developerKey=api_key,
                         discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                         static_discovery=False)


def perspective(temp_file):
    analyze_request = {'comment': {'text': temp_file},
                       'requestedAttributes': {
                            'TOXICITY':{}
                        },
                       'doNotStore':True,
                       'languages':['en']}
    response = client.comments().analyze(body=analyze_request).execute()
    return response


# Biased root-seeking random walk.
def weighted_random_walk(sentences, data, node_id, child_edges, walk_len, instance):
    length = 0
    sentences[0] = data['nodes'][node_id]['text']
    toxicity = data['nodes'][node_id]['toxicity']

    weight = 0.6
    for i in range(1, walk_len):
        length += 1
        choices = []
        probs = []
        if node_id in data['edges'] and data['edges'][node_id] != '':
            edge = data['edges'][node_id]
            if instance in data['nodes'][edge]['old_conv_ids']:
                choices.append(edge)
                probs.append(weight)
        if node_id in child_edges:
            # choices.extend(child_edges[node_id])
            num_child = len(child_edges[node_id])
            # probs.extend([(1-weight)/num_child]*num_child)

            for child_id in child_edges[node_id]:
                if instance in data['nodes'][child_id]['old_conv_ids']:
                    choices.append(child_id)
                    probs.append((1-weight)/num_child)

        if len(choices) == 0:
            return sentences, toxicity
        node = random.choices(choices, probs)[0]
        sentences[i] = data['nodes'][node]['text']
        node_id = node
    return sentences, toxicity


def start_random_walk(data, perspective_dict):
    child_edges = {}
    dataset_samples = []

    for node_id in data['nodes']:
        if node_id in data['edges'] and data['edges'][node_id] != '':
            parent_id = data['edges'][node_id]
            if parent_id in child_edges:
                child_edges[parent_id].append(node_id)
            else:
                child_edges[parent_id] = [node_id]
        #print(data['nodes'])

        # text = data['nodes'][node_id]['text'][:50]
        # if text in perspective_dict:
        #     data['nodes'][node_id]['toxicity'] = perspective_dict[text]['toxicity']
        if data['nodes'][node_id]['text'] != '':
            data['nodes'][node_id]['toxicity'] = perspective(data['nodes'][node_id]['text'].lower())['attributeScores']['TOXICITY']['summaryScore']['value']

    for node_id in data['nodes']:
        # for instance in list(data['nodes'][node_id]['old_conv_ids'].keys()):
        #if 'toxicity' in data['nodes'][node_id]:
        sentences = ['']*4
        sentences, toxicity = weighted_random_walk(sentences, data, node_id, child_edges, 4, 'shitposter.club')
        sentences.append(toxicity)
        dataset_samples.append(sentences)
    return dataset_samples



fp = open('perspective_dict.pkl', 'rb')
perspective_dict = pkl.load(fp)

instances = os.listdir('./instances_conv_graphs')
data_samples = []

# Medium instances: freethinkers.lgbt (25120) / a.nti.social (11560)

for instance in instances:
    if instance == 'shitposter.club':      # social.madeoutofmeat.com / my.dirtyhobby.xyz (x)
        convs = os.listdir('./instances_conv_graphs/' + instance)
        for conv_name in convs:
            if conv_name.startswith('http'):
                try:
                    with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                        conv = pkl.load(f)
                        if len(conv['nodes']) >= 5:
                            for toot_id in conv['nodes']:
                                #text = conv['nodes'][toot_id]['text'][:50]
                                #if text in perspective_dict:
                                    # if perspective_dict[text]['toxicity'] > 0.5:
                                data_samples.extend(start_random_walk(conv, perspective_dict))
                except:
                    pass


print('#samples:', len(data_samples))
pd.DataFrame(data_samples, columns=['sent1', 'sent2', 'sent3', 'sent4', 'toxicity']).to_csv('./pleroma_random_walks_inf.csv', index=False)
