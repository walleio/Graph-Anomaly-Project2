import pickle
from sklearn.metrics import roc_auc_score

with open('Jun27-10_examples-800_train.pkl', 'rb') as f:
    my_dict = pickle.load(f)

y_true = []
y_pred = []
for k, v in my_dict.items():
    if k == 'acc':
        continue
    y_true.append(v['label'])
    y_pred.append(1 if v['pred'] == 'Authentic' else 0)

print(roc_auc_score(y_true, y_pred))