
# Pipeline for constructing conversations out of toots.

import os
import json
import time
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt


def find_local_conv_id(instance, nodes):
    if os.path.exists('./pleroma_3/' + instance + '/toots/0/'):
        files = os.listdir('./pleroma_3/' + instance + '/toots/0/')

        for enum, file in enumerate(files):
            if not file.startswith('.'):
                try:
                    f = open('./pleroma_3/' + instance + '/toots/0/' + file, 'r', encoding='latin-1')
                    toots = json.load(f)
                    for toot in toots:
                        if toot['url'] in nodes:
                            return toot['pleroma']['conversation_id']
                except Exception:
                    return -1


def check_toots_in_other_instances(instances, nodes, edges):
    toot_id_url_map = {}
    conv_toots = []
    for ins in instances:
        # Find local conv id
        new_conv_id = find_local_conv_id(ins, nodes)

        if os.path.exists('./pleroma_3/' + ins + '/toots/0/') and new_conv_id != -1:
            files = os.listdir('./pleroma_3/' + ins + '/toots/0/')
            for enum, file in enumerate(files):
                if not file.startswith('.'):
                    try:
                        f = open('./pleroma_3/' + ins + '/toots/0/' + file, 'r', encoding='latin-1')
                        toots = json.load(f)

                        for toot in toots:
                            if toot['pleroma']['conversation_id'] == new_conv_id:
                                toot_id_url_map[toot['id']] = toot['url']
                                conv_toots.append(toot)
                    except Exception:
                        pass

            for toot in conv_toots:
                if toot['url'] in nodes:
                    nodes[toot['url']]['old_conv_ids'].update({ins.replace('/', ''): toot['pleroma']['conversation_id']})
                else:
                    nodes[toot['url']] = {
                        'created_at': toot['created_at'],
                        'text': toot['pleroma']['content']['text/plain'],
                        'author': toot['account']['acct'],
                        'instance': ins.replace('/', ''),
                        'reblogged': toot['reblogged'],
                        'reblogs_count': toot['reblogs_count'],
                        'old_conv_ids': {ins.replace('/', ''): toot['pleroma']['conversation_id']}
                    }
                    if toot['in_reply_to_id'] is not None and toot['in_reply_to_id'] in toot_id_url_map:
                        edges[toot['url']] = toot_id_url_map[toot['in_reply_to_id']]
    return nodes, edges


servers = os.listdir('./pleroma_3')
start_time = time.time()
for indx, server in enumerate(servers):
    conv_toot_map = {}
    toot_id_url_map = {}
    files = []

    if server != 'troll.cafe' and indx > 1110:
        print('Processing', indx, server, ' ...')

        if os.path.exists('./pleroma_3/' + server + '/toots/0/'):
            files = os.listdir('./pleroma_3/' + server + '/toots/0/')

        for enum, file in enumerate(files):
            if not file.startswith('.'):
                f = open('./pleroma_3/' + server + '/toots/0/' + file, 'r')
                toots = json.load(f)
                for toot in toots:
                    if 'pleroma' in toot:
                        conv_id = toot['pleroma']['conversation_id']
                        if conv_id not in conv_toot_map:
                            conv_toot_map[conv_id] = [toot]
                        else:
                            conv_toot_map[conv_id].append(toot)
                        toot_id_url_map[toot['id']] = toot['url']

        for enum, conv_id in enumerate(conv_toot_map):
            if enum >= 0:
                nodes = {}
                edges = {}
                instances = set()
                new_conv_id = ''

                for toot in conv_toot_map[conv_id]:
                    if len(toot['account']['acct'].split('@')) > 1:
                        instance = toot['account']['acct'].split('@')[1]
                        instances.add(instance)
                    else:
                        instance = server.replace('/', '')
                    nodes[toot['url']] = {
                        'created_at': toot['created_at'],
                        'text': toot['pleroma']['content']['text/plain'],
                        'author': toot['account']['acct'],
                        'instance': instance,
                        'reblogged': toot['reblogged'],
                        'reblogs_count': toot['reblogs_count'],
                        'old_conv_ids': {server.replace('/', ''): toot['pleroma']['conversation_id']}
                    }
                    if toot['in_reply_to_id'] is None:
                        edges[toot['url']] = ''
                        new_conv_id = toot['url']
                    else:
                        if toot['in_reply_to_id'] in toot_id_url_map:
                            edges[toot['url']] = toot_id_url_map[toot['in_reply_to_id']]

                # nodes, edges = check_toots_in_other_instances(list(instances), nodes, edges)

                path = './instances_conv_graphs/' + server + '/' + new_conv_id.replace('/', '\\') + '.pkl'
                os.makedirs(os.path.dirname(path), exist_ok=True)

                if len(path) <= 200:
                    with open(path, 'wb') as f:
                        pkl.dump({'nodes': nodes, 'edges': edges}, f)
        print('Processing time', time.time() - start_time)
