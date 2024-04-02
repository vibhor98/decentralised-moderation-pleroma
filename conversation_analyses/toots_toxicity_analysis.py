
import os
import time
import json
import math
import numpy as np
import pickle as pkl
from scipy import stats
import statistics
import powerlaw
import pandas as pd
import matplotlib.pyplot as plt

toots_per_conv = []
toxic_toots_per_conv = []
fed_toxic_toots = []
fed_nontoxic_toots = []
toxic_reblogs = []
nontoxic_reblogs = []
instances_conv = []
users_per_ins = []

ins_toxic_toots = {}
conv_len_toxic_toots = {}   # boxplot
reblogs_toxic_toots = {}    # barplot


with open('perspective_dict.pkl', 'rb') as f:
    perspective_dict = pkl.load(f)

instances = os.listdir('./instances_conv_graphs')

for instance in instances:
    toxic_toots_per_ins = 0
    unique_users = set()
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        toxic_toots = 0
        if conv_name.startswith('http'):
            try:
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    toots_per_conv.append(len(conv['nodes']))
                    conv_len = len(conv['nodes'])
                    instances_conv.append(instance)

                    for toot_id in conv['nodes']:
                        text = conv['nodes'][toot_id]['text'][:50]
                        unique_users.add(conv['nodes'][toot_id]['author'])
                        if text in perspective_dict:
                            if perspective_dict[text]['toxicity'] > 0.5:
                                toxic_toots += 1
                                toxic_toots_per_ins += 1
                                fed_toxic_toots.append(len(conv['nodes'][toot_id]['old_conv_ids']))
                                reblogs = conv['nodes'][toot_id]['reblogs_count']
                                toxic_reblogs.append(reblogs)
                                if reblogs in reblogs_toxic_toots:
                                    reblogs_toxic_toots[reblogs] += 1
                                else:
                                    reblogs_toxic_toots[reblogs] = 1

                                if conv_len not in conv_len_toxic_toots:
                                    conv_len_toxic_toots[conv_len] = []
                            else:
                                fed_nontoxic_toots.append(len(conv['nodes'][toot_id]['old_conv_ids']))
                                nontoxic_reblogs.append(conv['nodes'][toot_id]['reblogs_count'])
                    toxic_toots_per_conv.append(toxic_toots)
                    conv_len_toxic_toots[conv_len].append(toxic_toots)
            except:
                pass
    ins_toxic_toots[instance] = toxic_toots_per_ins
    users_per_ins.append(len(unique_users))

print('Avg. no. of toxic toots per conversation:', sum(toxic_toots_per_conv)/len(toxic_toots_per_conv))
print('Max no. of toxic toots per conversation:', max(toxic_toots_per_conv))
print('Median no. of toxic toots per conversation:', statistics.median(toxic_toots_per_conv))

print('Avg. no. of instances a toxic toot is federated to:', sum(fed_toxic_toots)/len(fed_toxic_toots))
print('Avg. no. of instances a non-toxic toot is federated to:', sum(fed_nontoxic_toots)/len(fed_nontoxic_toots))
print()
print('Max no. of instances a toxic toot is federated to:', max(fed_toxic_toots))
print('Max no. of instances a non-toxic toot is federated to:', max(fed_nontoxic_toots))

print('Avg. no. of reblogs for toxic toots:', sum(toxic_reblogs)/len(toxic_reblogs))
print('Avg. no. of reblogs for non-toxic toots:', sum(nontoxic_reblogs)/len(nontoxic_reblogs))
print()
print('Max no. of reblogs for toxic toots:', max(toxic_reblogs))
print('Max no. of reblogs for non-toxic toots:', max(nontoxic_reblogs))
print()
print('Median no. of reblogs for toxic toots:', statistics.median(toxic_reblogs))
print('Median no. of reblogs for non-toxic toots:', statistics.median(nontoxic_reblogs))
print()
print('Avg. no. of users per instance:', statistics.mean(users_per_ins))
print('Median no. of users per instance:', statistics.median(users_per_ins))
print('Max no. of users per instance:', max(users_per_ins))

df = pd.DataFrame(columns=['instance', 'toots_per_conv', 'toxic_toots_per_conv'])
df['instance'] = instances_conv
df['toots_per_conv'] = toots_per_conv
df['toxic_toots_per_conv'] = toxic_toots_per_conv

df.to_csv('num_toots_per_conv.csv', index=False)

# CDF of no. of toxic toots per conversation
plt.figure(0)
# plt.plot(sorted(toxic_toots_per_conv), stats.norm.cdf(sorted(toxic_toots_per_conv)))
tt_x, tt_y = powerlaw.cdf(toxic_toots_per_conv)
plt.plot(tt_x, tt_y)
plt.xlabel('No. of toxic toots per conversation')
plt.ylabel('Probability')
plt.savefig('cdf_toxic_toots_per_conv.png', bbox_inches='tight')

# CDF of no. of instances toxic/non-toxic toots are federated to
plt.figure(1)
# plt.plot(sorted(fed_toxic_toots), stats.norm.cdf(sorted(fed_toxic_toots)), label='Toxic Toots')
# plt.plot(sorted(fed_nontoxic_toots), stats.norm.cdf(sorted(fed_nontoxic_toots)), label='Non-toxic Toots')
tt_x, tt_y = powerlaw.cdf(fed_toxic_toots)
ntt_x, ntt_y = powerlaw.cdf(fed_nontoxic_toots)
plt.plot(ntt_x, ntt_y, label='Non-toxic Toots', color='orange')
plt.plot(tt_x, tt_y, label='Toxic Toots', color='blue')
plt.xlabel('No. of instances a toot is federated to')
plt.ylabel('Probability')
plt.legend()
plt.savefig('cdf_fed_ins_toxic.png', bbox_inches='tight')

# CDF of no. of reblogs for toxic/non-toxic toots
plt.figure(0)
tt_x, tt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in toxic_reblogs])
ntt_x, ntt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in nontoxic_reblogs])
plt.plot(ntt_x, ntt_y, label='Non-toxic Toots', color='orange')
plt.plot(tt_x, tt_y, label='Toxic Toots', color='blue')
plt.xlabel('log10 (Number of toot reblogs)')
plt.ylabel('Probability')
plt.legend()
plt.savefig('cdf_toot_reblogs_toxic.png', bbox_inches='tight')

# Barplot of top 10 instances with toxic toots
ins_toxic_toots = dict(sorted(ins_toxic_toots.items(), key=lambda item: item[1], reverse=True))
print('% toxic toots coming from mydirtyhobby.xyz :', list(ins_toxic_toots.values())[0] / sum(list(ins_toxic_toots.values())))
print('% toxic toots coming from', list(ins_toxic_toots.keys())[1], list(ins_toxic_toots.values())[1] / sum(list(ins_toxic_toots.values())))
print('% toxic toots coming from', list(ins_toxic_toots.keys())[2], list(ins_toxic_toots.values())[2] / sum(list(ins_toxic_toots.values())))
print('% toxic toots coming from', list(ins_toxic_toots.keys())[3], list(ins_toxic_toots.values())[3] / sum(list(ins_toxic_toots.values())))

plt.figure(1)
plt.bar(range(20), list(ins_toxic_toots.values())[:20])
plt.xticks(range(20), list(ins_toxic_toots.keys())[:20], rotation=90)
plt.xlabel('Instance')
plt.ylabel('Number of Toxic Toots')
plt.savefig('ins_toxic_toots_barplot.png', bbox_inches='tight')

# Boxplot of no. of toxic toots per conversation wrt conversation length

plt.figure(0)
plt.bar(list(conv_len_toxic_toots.keys()), [math.log10(sum(t)) for t in list(conv_len_toxic_toots.values())])
# plt.xticks(list(conv_len_toxic_toots.keys()))
plt.xlabel('Conversation Length')
plt.ylabel('No. of toxic toots per conversation')
plt.savefig('toxic_toots_conv_len_bar.png', bbox_inches='tight')

plt.figure(0)
plt.bar(list(conv_len_toxic_toots.keys()),
    [len([e for e in t if e>0]) / len(t)*100 for t in list(conv_len_toxic_toots.values())])
# plt.xticks(list(conv_len_toxic_toots.keys()))
plt.xlabel('Conversation Length')
plt.ylabel('Percentage of conversations with at least 1 toxic toot')
plt.savefig('toxic_conv_percent_conv_len_bar.png', bbox_inches='tight')


# Barplot of no. of toxic toots wrt reblogs
plt.figure(1)
plt.bar(list(reblogs_toxic_toots.keys()), list(reblogs_toxic_toots.values()))
plt.xlabel('Toot Reblogs')
plt.ylabel('Number of Toxic Toots')
plt.savefig('toxic_toots_reblogs_barplot.png', bbox_inches='tight')

############################################
# Compute toxic toots level in conversation trees

toxic_toot_position = {}
num_replies_toxic = []
num_replies_nontoxic = []
root_toot_id = ''

for instance in instances:
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        if conv_name.startswith('http'):
            with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                try:
                    conv = pkl.load(f)
                    parent_child_edges = {}
                    for edge in conv['edges']:
                        if conv['edges'][edge] == '':
                            root_toot_id = edge
                        else:
                            if conv['edges'][edge] in parent_child_edges:
                                parent_child_edges[conv['edges'][edge]].append(edge)
                            else:
                                parent_child_edges[conv['edges'][edge]] = [edge]

                        # Breadth-first Traversal
                        queue = [root_toot_id]

                        while len(queue) != 0:
                            toot_id = queue.pop(0)
                            if conv['edges'][toot_id] == '':
                                conv['nodes'][toot_id]['pos'] = 1
                            else:
                                conv['nodes'][toot_id]['pos'] = conv['nodes'][conv['edges'][toot_id]]['pos'] + 1

                            text = conv['nodes'][toot_id]['text'][:50]
                            if text in perspective_dict and perspective_dict[text]['toxicity'] > 0.5:
                                if toot_id in parent_child_edges:
                                    num_replies_toxic.append(len(parent_child_edges[toot_id]))
                                if conv['nodes'][toot_id]['pos'] not in toxic_toot_position:
                                    toxic_toot_position[conv['nodes'][toot_id]['pos']] = 1
                                else:
                                    toxic_toot_position[conv['nodes'][toot_id]['pos']] += 1
                            else:
                                if toot_id in parent_child_edges:
                                    num_replies_nontoxic.append(len(parent_child_edges[toot_id]))

                            if toot_id in parent_child_edges:
                                queue.extend(parent_child_edges[toot_id])
                except:
                    pass

# Barplot of no. of toxic toots wrt toot level
print(toxic_toot_position)

toxic_toot_position = {k: math.log10(v) for k, v in sorted(toxic_toot_position.items(), key=lambda item: item[0])}

plt.figure(0)
plt.bar(range(len(list(toxic_toot_position.keys()))), list(toxic_toot_position.values()))
plt.xticks(list(toxic_toot_position.keys()))
plt.xlabel('Toot Level in a Discussion Tree')
plt.ylabel('Number of Toxic Toots')
plt.savefig('toxic_toots_level_barplot.png', bbox_inches='tight')

print('Avg. no. of replies to toxic toots:', sum(num_replies_toxic)/len(num_replies_toxic))
print('Max no. of replies to toxic toots:', max(num_replies_toxic))
print('Median no. of replies to toxic toots:', statistics.median(num_replies_toxic))

print('Avg. no. of replies to non-toxic toots:', sum(num_replies_nontoxic)/len(num_replies_nontoxic))
print('Max no. of replies to non-toxic toots:', max(num_replies_nontoxic))
print('Median no. of replies to non-toxic toots:', statistics.median(num_replies_nontoxic))

plt.figure(1)
tt_x, tt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in num_replies_toxic])
ntt_x, ntt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in num_replies_nontoxic])
plt.plot(ntt_x, ntt_y, label='Non-toxic Toots', color='orange')
plt.plot(tt_x, tt_y, label='Toxic Toots', color='blue')
plt.xlabel('log10 (Number of replies to a toot)')
plt.ylabel('Probability')
plt.legend()
plt.savefig('cdf_num_replies_toxic.png', bbox_inches='tight')
