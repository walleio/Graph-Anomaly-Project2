import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import itertools
from torch_geometric.data import HeteroData, Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html

#### node processing ###
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
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
    df = pd.read_csv(path, **kwargs)

    edges = set()

    # filter to connect nodes with that have reviewed the same product
    grouped = df.groupby('productId')

    for productId, group in grouped:
        user_list = group['userId']

        for u1, u2 in itertools.combinations(user_list, 2):
            edges.add(tuple((u1, u2)))    

    # filter to connect nodes that have given the same score
    grouped = df.groupby('score')

    for score, group in grouped:
        user_list = []
        for x, i in enumerate(group['time']):
            for y, j in enumerate(group['time'][1:]):
                if (abs(i - j) < 604800):
                    edges.add(tuple((group['userId'].iloc[x], group['userId'].iloc[y])))

    # filter for tf-idf
    df_per_user = (
        df.groupby("userId", sort=False)["text"]
        .apply(" ".join)
        .reset_index()
    )

    vectoriser = TfidfVectorizer(
        min_df=3,            
        max_df=0.8,        
        ngram_range=(1, 2),  
    )

    tfidf_matrix = vectoriser.fit_transform(df_per_user["text"].values)

    N = tfidf_matrix.shape[0]
    k = max(1, int(0.05 * (N - 1)))

    S = cosine_similarity(tfidf_matrix, dense_output=True) 
    user_ids = df_per_user["userId"].to_numpy()

    for u in range(N):
        top_k = S[u].argsort()[::-1][1 : k + 1]
        for v in top_k:
            if u in S[v].argsort()[::-1][1 : k + 1]:
                edges.add(tuple(sorted((user_ids[u], user_ids[v]))))

    src = []
    dst = []
    for a,b in edges:
        src.append(mapping[a])
        dst.append(mapping[b])

    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr
    
def process_data():
    x, mapping = load_node_csv('~/Graph-Anomaly-Project/data/users.csv', 'userId', encoders={
    'profileName': SequenceEncoder()})


    edge_index, edge_attrs = load_edge_csv('~/Graph-Anomaly-Project/data/reviews.csv', mapping, encoders={
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
    labels_df = pd.read_csv('~/Graph-Anomaly-Project/data/users.csv')
    data.y = torch.tensor(labels_df['label'])

    data.edge_index = edge_index
    data.edge_labels = edge_attrs

    return data
