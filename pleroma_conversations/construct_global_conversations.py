
# Pipeline to construct global conversations out of local conversations.

import os
import time
import pickle as pkl

instances = os.listdir('./instances_conv_graphs/')


for indx, instance in enumerate(instances):
    start_time = time.time()
    if indx >= 10:
        print('Processing', indx, instance)
        conv_files = os.listdir('./instances_conv_graphs/' + instance)
        for file in conv_files[1:]:
            if not file.startswith('.'):
                f = open('./instances_conv_graphs/' + instance + '/' + file, 'rb')
                conv = pkl.load(f)
                fed_instances = set([conv['nodes'][toot]['instance'] for toot in conv['nodes'] if conv['nodes'][toot]['instance'] != instance])

                for fed_ins in fed_instances:
                    if os.path.exists('./instances_conv_graphs/' + fed_ins + '/' + file):
                        print('Path exists', './instances_conv_graphs/' + fed_ins + '/' + file)
                        fed_f = open('./instances_conv_graphs/' + fed_ins + '/' + file, 'rb')
                        fed_conv = pkl.load(fed_f)
                        for toot_id in fed_conv['nodes']:
                            if toot_id in conv['nodes']:
                                conv['nodes'][toot_id]['old_conv_ids'].update({fed_ins: ''})
                            else:
                                conv['nodes'][toot_id] = fed_conv['nodes'][toot_id]
                                if toot_id in fed_conv['edges']:
                                    conv['edges'][toot_id] = fed_conv['edges'][toot_id]

                with open('./instances_conv_graphs/' + instance + '/' + file, 'wb') as f:
                    pkl.dump(conv, f)

        print('Processing time', time.time() - start_time)
