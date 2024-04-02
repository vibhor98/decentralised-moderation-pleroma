
# Pipeline for processing user data.

import os
import time
import json
import math
import pandas as pd
import numpy as np
import pickle as pkl
from scipy import stats
import statistics
import powerlaw
import matplotlib.pyplot as plt
from collections import Counter


# instances = os.listdir('./pleroma_3')
start_time = time.time()
user_data = {}
sensitive_data = {}

for indx, instance in enumerate(instances):
    files = []

    print('Processing', indx, instance, ' ...')

    if os.path.exists('./pleroma_3/' + instance + '/toots/0/'):
        files = os.listdir('./pleroma_3/' + instance + '/toots/0/')

        for enum, file in enumerate(files):
            if not file.startswith('.'):
                try:
                    f = open('./pleroma_3/' + instance + '/toots/0/' + file, 'r')
                    toots = json.load(f)
                    for toot in toots:
                        # if toot['account']['acct'] not in user_data:
                        #     user_data[toot['account']['acct']] = {
                        #         'followers': toot['account']['followers_count'],
                        #         'following': toot['account']['following_count']
                        #     }
                        if toot['url'] not in sensitive_data:
                            sensitive_data[toot['url']] = {
                                'sensitive': toot['sensitive'],
                                'spoiler_text': toot['spoiler_text']
                            }
                except:
                    pass

# with open('user_data.pkl', 'wb') as f:
#     pkl.dump(user_data, f)
with open('sensitive_data.pkl', 'wb') as f:
    pkl.dump(sensitive_data, f)

print('Time taken:', time.time()-start_time)

###############

sensitive_toots_per_conv = []
fed_sens_toots = []
fed_nonsens_toots = []

with open('sensitive_data.pkl', 'rb') as f:
    sensitivity_dict = pkl.load(f)

instances = os.listdir('./instances_conv_graphs')

for instance in instances:
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        sensitive_toots = 0
        if conv_name.startswith('http'):
            try:
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    num_toots = len(conv['nodes'])

                    for toot_id in conv['nodes']:
                        if toot_id in sensitivity_dict:
                            if sensitivity_dict[toot_id]['sensitive']:    #True
                                sensitive_toots += 1
                                fed_sens_toots.append(len(conv['nodes'][toot_id]['old_conv_ids']))
                            else:
                                fed_nonsens_toots.append(len(conv['nodes'][toot_id]['old_conv_ids']))
                sensitive_toots_per_conv.append(sensitive_toots)
            except:
                pass

print('Avg. no. of sensitive toots per conversation:', sum(sensitive_toots_per_conv)/len(sensitive_toots_per_conv))
print('Max no. of sensitive toots per conversation:', max(sensitive_toots_per_conv))
print('Min no. of sensitive toots per conversation:', min(sensitive_toots_per_conv))

print('Avg. no. of instances a sensitive toot is federated with:', sum(fed_sens_toots)/len(fed_sens_toots))
print('Avg. no. of instances a non-sensitive toot is federated with:', sum(fed_nonsens_toots)/len(fed_nonsens_toots))
print()
print('Max no. of instances a sensitive toot is federated with:', max(fed_sens_toots))
print('Max no. of instances a non-sensitive toot is federated with:', max(fed_nonsens_toots))

###############

instances = os.listdir('./instances_conv_graphs/')
ins_toots = []
print('No. of instances:', len(instances))

num_convs_per_ins = []
num_toots_per_conv = []
num_toots_per_conv2 = []
num_convs_more_than_1_toot = 0

for instance in instances:
    num_users = set()
    convs = os.listdir('./instances_conv_graphs/' + instance)
    num_convs_per_ins.append(len(convs))
    for conv_name in convs:
        if conv_name.startswith('http'):
            try:
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    if len(conv['nodes']) > 1:
                        num_convs_more_than_1_toot += 1
                    if len(conv['nodes']) > 2:
                        num_toots_per_conv2.append(len(conv['nodes']))
                    num_toots_per_conv.append(len(conv['nodes']))
            except:
                pass

print('Avg. no. of conversations per instance:', sum(num_convs_per_ins) / len(num_convs_per_ins))
print('Max no. of conversations per instance:', max(num_convs_per_ins))

print('Avg. no. of toots per conversation:', sum(num_toots_per_conv) / len(num_toots_per_conv))
print('Max no. of toots per conversation:', max(num_toots_per_conv))
print('Median no. of toots per conversation:', statistics.median(num_toots_per_conv))

print('Avg. no. of toots per conversation:', sum(num_toots_per_conv2) / len(num_toots_per_conv2))
print('Max no. of toots per conversation:', max(num_toots_per_conv2))
print('Median no. of toots per conversation:', statistics.median(num_toots_per_conv2))

print('% conversations with more than 1 toot:', num_convs_more_than_1_toot/sum(num_convs_per_ins))

# y-axis is on log scale
plt.figure(0)
df = pd.DataFrame()
df['convs'] = num_convs_per_ins
# df['convs'].hist(bins=20, log=True)
df['convs'].plot(kind='hist', bins=200, logx=True, logy=True)
plt.xlabel('log10 (No. of conversations per instance)')
plt.ylabel('log10 (Number of instances)')
plt.grid()
plt.savefig('num_conv_per_ins_log.png', bbox_inches='tight')

plt.figure(1)
df = pd.DataFrame()
df['toots'] = num_toots_per_conv
# df['toots'].hist(bins=20, log=True)
df['toots'].plot(kind='hist', bins=200, logx=True, logy=True)
plt.xlabel('log10 (No. of toots per conversation)')
plt.ylabel('log10 (Number of conversations)')
plt.grid()
plt.savefig('num_toots_per_conv_log.png', bbox_inches='tight')

plt.figure(2)
# plt.plot(sorted(num_convs_per_ins), stats.norm.cdf(sorted(num_convs_per_ins)))
tt_x, tt_y = powerlaw.cdf(sorted(num_convs_per_ins))
plt.plot(tt_x, tt_y, color='blue')
plt.xlabel('No. of conversations per instance')
plt.ylabel('Probability')
plt.savefig('cdf_conv_per_ins.png', bbox_inches='tight')

plt.figure(3)
# plt.plot(sorted(num_toots_per_conv), stats.norm.cdf(sorted(num_toots_per_conv)))
tt_x, tt_y = powerlaw.cdf(sorted(num_toots_per_conv))
plt.plot(tt_x, tt_y, color='blue')
plt.xlabel('No. of toots per conversation')
plt.ylabel('Probability')
plt.savefig('cdf_toots_per_conv.png', bbox_inches='tight')

#############

fed_ins_per_ins = []
fed_ins_per_conv = []
fed_ins_per_conv2 = []
local_toots = []
fed_toots = []

local_toots_per_conv = []
fed_toots_per_conv = []

toot_popularity = []
toot_federation1 = []
toot_federation2 = []
user_followers = []
user_following = []

conv_len = []
ins_size = []

with open('user_data.pkl', 'rb') as f:
    followers_dict = pkl.load(f)

for instance in instances:
    ins_set = set()
    local_t = 0
    fed_t = 0
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        ins_set_per_conv = set()
        local_t_per_conv = 0
        fed_t_per_conv = 0
        if conv_name.startswith('http'):
            try:
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    for toot_id in conv['nodes']:
                        ins_set_per_conv.update(list(conv['nodes'][toot_id]['old_conv_ids'].keys()))

                        if conv['nodes'][toot_id]['reblogs_count'] > 10:    # Filtering popular toots
                            toot_popularity.append(conv['nodes'][toot_id]['reblogs_count'])
                            toot_federation1.append(len(conv['nodes'][toot_id]['old_conv_ids']))

                        # if conv['nodes'][toot_id]['author'] in followers_dict:
                        # if conv['edges'][toot_id] == '':    # Considering only root nodes
                        user_followers.append(followers_dict[conv['nodes'][toot_id]['author']]['followers'])
                        user_following.append(followers_dict[conv['nodes'][toot_id]['author']]['following'])
                        toot_federation2.append(len(conv['nodes'][toot_id]['old_conv_ids']))

                        if conv['nodes'][toot_id]['instance'] == instance:
                            local_t += 1
                            local_t_per_conv += 1
                        else:
                            fed_t += 1
                            fed_t_per_conv += 1
                    if len(conv['nodes']) > 2:
                        fed_ins_per_conv2.append(len(ins_set_per_conv))
                    fed_ins_per_conv.append(len(ins_set_per_conv))
                    conv_len.append(len(conv['nodes']))
            except:
                pass
        ins_set.update(ins_set_per_conv)
        local_toots_per_conv.append(local_t_per_conv)
        fed_toots_per_conv.append(fed_t_per_conv)
    fed_ins_per_ins.append(len(ins_set))
    ins_size.append(fed_t)   # + local_t
    if local_t != 0:
        local_t = math.log10(local_t)
    local_toots.append(local_t)
    if fed_t != 0:
        fed_t = math.log10(fed_t)
    fed_toots.append(fed_t)

# Avg. no. of federated instances per instance
print('Avg. no. of instances an instance is federating with:', sum(fed_ins_per_ins) / len(fed_ins_per_ins))
print('Max no. of instances an instance is federating with:', max(fed_ins_per_ins))
print('Median no. of instances an instance is federating with:', statistics.median(fed_ins_per_ins))

# Avg. no. of federated instances per conversation
print('Avg. no. of instances to which a conversation is federated:', sum(fed_ins_per_conv2) / len(fed_ins_per_conv2))
print('Max no. of instances to which a conversation is federated:', max(fed_ins_per_conv2))
print('Median no. of instances to which a conversation is federated:', statistics.median(fed_ins_per_conv2))
print(Counter(fed_ins_per_conv))

# Avg. no. of local and federated toots per conversation
print('Avg. no. of local toots per conversation:', sum(local_toots_per_conv) / len(local_toots_per_conv))
print('Avg. no. of federated toots per conversation:', sum(fed_toots_per_conv) / len(fed_toots_per_conv))

print('Median no. of local toots per conversation:', statistics.median(local_toots_per_conv))
print('Median no. of federated toots per conversation:', statistics.median(fed_toots_per_conv))

# CDF of local vs federated toots per conversation
plt.figure(0)
# plt.plot(stats.norm.cdf(sorted(local_toots_per_conv)), label='Local')
# plt.plot(stats.norm.cdf(sorted(fed_toots_per_conv)), label='Federated')
tt_x, tt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in list(sorted(local_toots_per_conv))])
ntt_x, ntt_y = powerlaw.cdf([math.log10(i) if i!=0 else 0 for i in list(sorted(fed_toots_per_conv))])
plt.plot(ntt_x, ntt_y, label='Federated Toots', color='orange')
plt.plot(tt_x, tt_y, label='Local Toots', color='blue')
plt.xlabel('No. of toots per conversation')
plt.ylabel('Probability')
plt.legend()
plt.savefig('cdf_local_fed_toots.png', bbox_inches='tight')

# CDF of no. of instances per conversation
plt.figure(1)
# plt.plot(sorted(fed_ins_per_conv), stats.norm.cdf(sorted(fed_ins_per_conv)))
tt_x, tt_y = powerlaw.cdf(sorted(fed_ins_per_conv))
plt.plot(tt_x, tt_y, color='blue')
plt.xlabel('No. of instances per conversation')
plt.ylabel('Probability')
plt.savefig('cdf_num_ins_per_conv.png', bbox_inches='tight')

plt.figure(1)
plt.scatter(toot_popularity, toot_federation1)
plt.xlabel('Toot Popularity (No. of Toot Reblogs)')
plt.ylabel('No. of Instances a Toot is federated with')
plt.savefig('toot_popularity_vs_fed.png', bbox_inches='tight')

plt.figure(2)
plt.scatter(conv_len, fed_ins_per_conv)
plt.xlabel('Conversation Length (No. of Toots)')
plt.ylabel('No. of Instances a Conversation is federated with')
plt.savefig('conv_len_vs_fed.png', bbox_inches='tight')

Instance size based on the no. of local toots only
plt.figure(3)
plt.scatter(ins_size, [math.log10(i) for i in fed_ins_per_ins])
plt.xlabel('Instance Size (No. of Local Toots)')
plt.ylabel('No. of Instances an Instance is federating with')
plt.savefig('ins_size_vs_fed.png', bbox_inches='tight')

plt.figure(1)
plt.scatter(ins_size, [math.log10(i) for i in fed_ins_per_ins])
plt.xlabel('Instance Size (No. of Federated Toots)')
plt.ylabel('No. of Instances an Instance is federating with')
plt.savefig('incoming_toots_vs_fed.png', bbox_inches='tight')

plt.figure(4)
plt.scatter(user_followers, toot_federation2)
plt.xlabel("No. of users' followers")
plt.ylabel('No. of Instances their Toot is federated with')
plt.savefig('users_followers_vs_fed.png', bbox_inches='tight')

plt.figure(5)
plt.scatter(user_following, toot_federation2)
plt.xlabel("No. of users' following")
plt.ylabel('No. of Instances their Toot is federated with')
plt.savefig('users_following_vs_fed.png', bbox_inches='tight')

########

What % of the conversation is present in a given instance?
num_ins_percent = [[] for i in range(15)]

for instance in instances:
    convs = os.listdir('./instances_conv_graphs/' + instance)
    for conv_name in convs:
        num_toots = 0
        ins_toots_dict = {}
        if conv_name.startswith('http'):
            try:
                with open('./instances_conv_graphs/' + instance + '/' + conv_name, 'rb') as f:
                    conv = pkl.load(f)
                    num_toots = len(conv['nodes'])
                    if num_toots > 1:
                        for toot_id in conv['nodes']:
                            for fed_ins in conv['nodes'][toot_id]['old_conv_ids']:
                                if fed_ins not in ins_toots_dict:
                                    ins_toots_dict[fed_ins] = 1
                                else:
                                    ins_toots_dict[fed_ins] += 1
                        d = dict((k, v/num_toots * 100) for (k, v) in ins_toots_dict.items())
                        num_ins_percent[len(d)].extend(list(d.values()))
            except:
                pass

num_ins_percent.pop(0)
num_ins_percent.pop(-2)

plt.figure(0)
plt.boxplot(num_ins_percent)
plt.xticks(ticks=range(1, 14), labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, '13+'])
plt.xlabel('No. of federated instances')
plt.ylabel('% of toots in a conversation (Partial Federation)')
plt.savefig('toot_percent_boxplot.png', bbox_inches='tight')
