import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import itertools
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import remove_self_loops

#### node processing ###
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, nrows=1000, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

class SequenceEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x

class IdentityEncoder:
    def __init__(self, dtype=torch.long):
        self.dtype = dtype
        self.device = 'cuda'

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype).to(self.device)

def load_edge_csv(path, mapping, encoders=None, device='cuda', **kwargs):
    df = pd.read_csv(path, nrows=1000, **kwargs)

    edges = []

    # filter to connect nodes that have reviewed the same product
    grouped = df.groupby('productId')

    for productId, group in grouped:
        user_list = group['userId']

        for u1, u2 in itertools.combinations(user_list, 2):
            edges.append((u1, u2))

    # filter to connect nodes that have given the same star rating in the same week
    grouped = df.groupby('score')  

    for score, group in grouped:
        user_list = []
        for x, i in enumerate(group['time']):
            for y, j in enumerate(group['time'][x+1:]):
                if (abs(i - j) < 604800):
                    edges.append((group['userId'].iloc[x], group['userId'].iloc[y+x+1]))

    #TODO: add TF-IDF to the edge attributes

    src = []
    dst = []
    for a, b in edges:
        src.append(mapping[a])
        dst.append(mapping[b])

    src_filtered = []
    dst_filtered = []
    for a, b in zip(src, dst):
        if a in dst_filtered and src_filtered[dst_filtered.index(a)] == b:
            continue
        else:
            src_filtered.append(a)
            dst_filtered.append(b)

    src_undirected = []
    dst_undirected = []
    for a, b in zip(src_filtered, dst_filtered):
        src_undirected.append(a)
        dst_undirected.append(b)
        src_undirected.append(b)
        dst_undirected.append(a)

    edge_index = torch.tensor([src_undirected, dst_undirected])

    '''
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    '''
    edge_attr = torch.tensor([], device='cuda')

    return edge_index, edge_attr
    
def process_data():
    x, mapping = load_node_csv('~/Graph-Anomaly-Project2/data/users3.csv', 'userId', encoders={
    'profileName': SequenceEncoder(),
    'review_example': SequenceEncoder()})

    edge_index, edge_attrs = load_edge_csv('~/Graph-Anomaly-Project2/data/reviews3.csv', mapping, encoders={
    'time': IdentityEncoder(),
    'score': IdentityEncoder(),
    'helpfulness numerator': IdentityEncoder(),
    'helpfulness denominator': IdentityEncoder(),
    'summary': SequenceEncoder(),
    'text': SequenceEncoder(),
    'productId': SequenceEncoder()})

    data = Data()
    data.num_nodes = len(mapping)
    data.x = x

    labels_df = pd.read_csv('~/Graph-Anomaly-Project2/data/users3.csv', nrows=1000)
    data.y = torch.tensor(labels_df["label"])
    
    edge_index, _ = remove_self_loops(edge_index)
    data.edge_index = edge_index

    return data
