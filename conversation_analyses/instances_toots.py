
# Counts number of toots in each instance

import os
import json
import pandas as pd
import pickle as pkl

# instances = os.listdir('./pleroma_3')
instances = os.listdir('./instances_conv_graphs/')
ins_toots = []


for enum, instance in enumerate(instances):
    print('Processing', enum, instance)

    if instance == 'www.banepo.st':
        if os.path.exists('./pleroma_3/' + instance + '/toots/0/'):
            files = os.listdir('./pleroma_3/' + instance + '/toots/0/')

            for file in files:
                if not file.startswith('.'):
                    f = open('./pleroma_3/' + instance + '/toots/0/' + file, 'r')
                    toots = json.load(f)
                    for toot in toots:
                        ins_toots.add(json.dumps(toot))

            toots = len(files) * 40
            ins_toots.append([instance, toots])

df = pd.DataFrame(ins_toots, columns=['instance', 'toots'])
df.to_csv('instance_toots.csv', index=False)
# print(len(ins_toots))

Total no. of conversations
num_toots = []

for instance in instances:
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        if conv_name.startswith('http'):
            with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                conv = pkl.load(f)
                num_toots.append(len(conv['nodes']))

print(len(num_toots))
print(sum(num_toots) / len(num_toots))
print(max(num_toots))

# Federated instances for a given instance
fed_ins_toots_dict = {}

for instance in instances:
    if instance == 'freethinkers.lgbt':      # pl.nudie.social / shigusegubu.club
        convs = os.listdir('./instances_conv_graphs/' + instance)
        for conv_name in convs:
            if conv_name.startswith('http'):
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    for toot_id in conv['nodes']:
                        for fed_ins in conv['nodes'][toot_id]['old_conv_ids']:
                            if fed_ins not in fed_ins_toots_dict:
                                fed_ins_toots_dict[fed_ins] = 1
                            else:
                                fed_ins_toots_dict[fed_ins] += 1

print(fed_ins_toots_dict)
