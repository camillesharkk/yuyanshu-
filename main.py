import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import random
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

# ---------------------------
# 1. 加载数据
# ---------------------------
def load_movielens_data():
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='ISO-8859-1', header=None)
    movies = movies[[0, 1, 5]]
    movies.columns = ['movieId', 'title', 'genres']
    movies['title'] = movies['title'].astype(str)
    movies['genres'] = movies['genres'].astype(str)
    return movies

# ---------------------------
# 2. 构建图（多种嵌入）
# ---------------------------
def build_item_graph(movies, mode='tfidf', sim_threshold=0.6):
    texts = [f"{t} {g}" for t, g in zip(movies['title'], movies['genres'])]
    if mode == 'tfidf':
        tfidf = TfidfVectorizer(stop_words='english')
        embeddings = tfidf.fit_transform(texts).toarray()
    elif mode == 'lsa':
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=100, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
    elif mode == 'bert' and BERT_AVAILABLE:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True)
    else:
        raise ValueError("Unsupported mode or BERT not available")

    sim_matrix = cosine_similarity(embeddings)
    edge_index = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i][j] > sim_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index).t().contiguous()
    x = torch.tensor(embeddings, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data, embeddings

# ---------------------------
# 3. GCN 模型与对比损失
# ---------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def contrastive_loss(embeddings, edge_index, temperature=0.5):
    num_nodes = embeddings.shape[0]
    loss = 0
    for i in range(num_nodes):
        pos_idx = edge_index[1][edge_index[0] == i]
        if len(pos_idx) == 0:
            continue
        anchor = embeddings[i]
        positives = embeddings[pos_idx]
        sim_pos = F.cosine_similarity(anchor.unsqueeze(0), positives).mean()

        all_idx = torch.arange(num_nodes)
        neg_idx = list(set(all_idx.tolist()) - set(pos_idx.tolist()) - {i})
        if len(neg_idx) == 0:
            continue
        neg_sample = embeddings[random.sample(neg_idx, min(5, len(neg_idx)))]
        sim_neg = F.cosine_similarity(anchor.unsqueeze(0), neg_sample).mean()

        loss += -torch.log(torch.exp(sim_pos / temperature) /
                          (torch.exp(sim_pos / temperature) + torch.exp(sim_neg / temperature)))
    return loss / num_nodes

# ---------------------------
# 4. 推荐与评估
# ---------------------------
def generate_recommendations(embeddings, user_histories, top_k=10):
    recommendations = {}
    for user_id, history in user_histories.items():
        user_vector = embeddings[history].mean(dim=0)
        sims = cosine_similarity(user_vector.view(1, -1), embeddings)[0]
        top_items = np.argsort(-sims)[:top_k]
        recommendations[user_id] = top_items
    return recommendations

def evaluate_ndcg(recommendations, ground_truth):
    ndcgs = []
    for user_id in recommendations:
        y_true = [1 if i in ground_truth[user_id] else 0 for i in range(len(recommendations[user_id]))]
        y_score = [1 / (rank + 1) for rank in range(len(recommendations[user_id]))]
        ndcg = ndcg_score([y_true], [y_score])
        ndcgs.append(ndcg)
    return np.mean(ndcgs)

# ---------------------------
# 5. 可视化
# ---------------------------
def visualize_embeddings(embeddings, genres, title="t-SNE of item embeddings"):
    print("Visualizing embeddings...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    genre_labels = list(set(genres))
    genre_to_color = {g: i for i, g in enumerate(genre_labels)}
    colors = [genre_to_color[g] for g in genres]

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap='tab20', s=6, alpha=0.7)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=plt.cm.tab20(genre_to_color[label]/20), markersize=5)
               for label in genre_labels[:10]]
    plt.legend(handles=handles, title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

# ---------------------------
# 6. 主消融实验流程
# ---------------------------
def run_experiment(mode):
    print(f"\n==== Embedding Mode: {mode} ====")
    movies = load_movielens_data()
    genres = movies['genres'].tolist()
    data, raw_embeddings = build_item_graph(movies, mode=mode)
    model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(51):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = contrastive_loss(out, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    item_embeddings = model(data.x, data.edge_index).detach()
    user_histories = {i: random.sample(range(len(movies)), 5) for i in range(10)}
    ground_truth = {i: random.sample(range(len(movies)), 3) for i in range(10)}
    recommendations = generate_recommendations(item_embeddings, user_histories)
    ndcg = evaluate_ndcg(recommendations, ground_truth)
    print(f"NDCG@10 for {mode}: {ndcg:.4f}")

    visualize_embeddings(item_embeddings.numpy(), genres, title=f"t-SNE of learned embeddings ({mode})")
    visualize_embeddings(raw_embeddings, genres, title=f"t-SNE of raw content embeddings ({mode})")
    return ndcg, loss.item()

# ---------------------------
# 7. 执行多种嵌入对比
# ---------------------------
if __name__ == '__main__':
    modes = ['tfidf', 'lsa']
    if BERT_AVAILABLE:
        modes.append('bert')
    results = []
    for m in modes:
        ndcg, loss_val = run_experiment(m)
        results.append((m, ndcg, loss_val))

    print("\n==== Summary (Embedding vs NDCG@10) ====")
    for m, n, l in results:
        print(f"{m.upper():10s} | NDCG@10: {n:.4f} | Final Loss: {l:.4f}")
