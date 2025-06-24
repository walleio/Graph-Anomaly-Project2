import pickle

# from a file -------------------------------------------------
with open('data/graph_pickle.pkl', 'rb') as f:
    my_dict = pickle.load(f)   # ← your dict is now in memory

map_dict = my_dict['map_dict']
labels = my_dict['labels']

for k, v in map_dict.items():
    print(f'target: {labels[k]}, examples: {labels[v[0]]}, {labels[v[1]]}, {labels[v[2]]}')
