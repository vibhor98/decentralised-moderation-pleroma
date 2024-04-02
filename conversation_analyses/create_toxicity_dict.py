
import pandas as pd
import pickle as pkl

data = pd.read_csv('perspective.csv.gz', compression='gzip', error_bad_lines=False)
perspective_dict = {}

for _, row in data.iterrows():
    perspective_dict[row['pleroma.content.text/plain'][:50]] = {
        'toot_id': row['id'],
        'toxicity': row['pleroma.content.toxicity'],
        'severe_toxicity': row['pleroma.content.severe_toxicity'],
        'is_local': row['pleroma.local'],
        'replies_count': row['replies_count'],
        'muted': row['muted'],
        'pinned': row['pinned'],
        'mentions': row['mentions']
    }
    if _ % 1000 == 0:
        print('Processed', _, 'rows...')

with open('perspective_dict.pkl', 'wb') as f:
    pkl.dump(perspective_dict, f)
