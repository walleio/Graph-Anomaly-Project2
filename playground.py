import pickle

# from a file -------------------------------------------------
with open('data/results.pkl', 'rb') as f:
    my_dict = pickle.load(f)   # ← your dict is now in memory

print(my_dict)