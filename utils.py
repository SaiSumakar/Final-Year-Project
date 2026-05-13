import os
import json
import random
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Vocabulary ─────────────────────────────────────────────────────────────────
def build_vocab(news_file: str, max_title_len: int, max_vocab_size: int):
    """
    Reads news.tsv and builds a word -> index vocabulary from titles.
    Returns:
        word2idx  : dict  {word: int}  (PAD=0, UNK=1)
        news_data : dict  {news_id: {'title': [word, ...], 'entities': [wikidata_id, ...]}}
    """
    from collections import Counter
    counter = Counter()
    raw = {}

    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            news_id   = parts[0]
            title     = parts[3].lower().split()[:max_title_len]
            try:
                entities = [e["WikidataId"] for e in json.loads(parts[6])]
            except Exception:
                entities = []
            counter.update(title)
            raw[news_id] = {"title": title, "entities": entities}

    # Keep top-N words
    most_common = [w for w, _ in counter.most_common(max_vocab_size - 2)]
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    for w in most_common:
        word2idx[w] = len(word2idx)

    return word2idx, raw


def merge_news_data(*news_dicts):
    """Merge multiple news_data dicts (train + val)."""
    merged = {}
    for d in news_dicts:
        merged.update(d)
    return merged


# ── Embeddings ─────────────────────────────────────────────────────────────────
def load_entity_embeddings(entity_file: str, entity_dim: int):
    """
    Returns:
        entity2idx  : dict {wikidata_id: int}  (PAD=0, UNK=1)
        emb_matrix  : np.ndarray  [num_entities, entity_dim]
    """
    entity2idx = {"[PAD]": 0, "[UNK]": 1}
    vectors = [np.zeros(entity_dim), np.zeros(entity_dim)]  # PAD, UNK

    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < entity_dim + 1:
                continue
            eid = parts[0]
            vec = np.array(parts[1:entity_dim + 1], dtype=np.float32)
            entity2idx[eid] = len(entity2idx)
            vectors.append(vec)

    emb_matrix = np.stack(vectors, axis=0).astype(np.float32)
    return entity2idx, emb_matrix



def build_word_embeddings(word2idx: dict, word_dim: int, glove_file: str = None):
    """
    Load word embeddings from GloVe file.
    For words not in GloVe, initialize with Xavier uniform distribution.
    PAD token (index 0) is set to all zeros.
    Returns np.ndarray [vocab_size, word_dim].
    """
    vocab_size = len(word2idx)
    emb = np.zeros((vocab_size, word_dim), dtype=np.float32)
    
    # Load GloVe embeddings if file is provided
    loaded_count = 0
    if glove_file and os.path.exists(glove_file):
        with open(glove_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < word_dim + 1:
                    continue
                word = parts[0]
                if word in word2idx:
                    try:
                        vec = np.array(parts[1:word_dim + 1], dtype=np.float32)
                        emb[word2idx[word]] = vec
                        loaded_count += 1
                    except (ValueError, IndexError) as e:
                        # Skip lines with malformed embeddings
                        continue
        print(f"  Loaded {loaded_count} / {vocab_size} words from GloVe")
    else:
        print(f"  GloVe file not found: {glove_file}")
    
    # For UNK and out-of-vocabulary words, initialize with Xavier uniform
    limit = np.sqrt(1.0 / vocab_size)
    for idx in range(1, vocab_size):  # skip PAD (index 0)
        if np.allclose(emb[idx], 0.0):  # not loaded from GloVe
            emb[idx] = np.random.uniform(-limit, limit, word_dim).astype(np.float32)
    
    emb[0] = 0.0   # PAD -> zero vector
    return emb


# ── Knowledge Graph ────────────────────────────────────────────────────────────
def build_entity_graph(news_data: dict, entity2idx: dict, max_neighbors: int):
    """
    Build entity co-occurrence graph from news.
    Two entities are connected if they appear in the same news article.

    Returns:
        adj : dict {entity_idx: [neighbor_entity_idx, ...]}  (truncated to max_neighbors)
    """
    co_occur = defaultdict(set)

    for info in news_data.values():
        eids = [entity2idx.get(e, 1) for e in info["entities"] if e in entity2idx]
        eids = [e for e in eids if e > 1]   # skip PAD/UNK
        for i in range(len(eids)):
            for j in range(len(eids)):
                if i != j:
                    co_occur[eids[i]].add(eids[j])

    adj = {}
    for eid, neighbors in co_occur.items():
        nbrs = list(neighbors)
        if len(nbrs) > max_neighbors:
            nbrs = random.sample(nbrs, max_neighbors)
        adj[eid] = nbrs

    return adj


def adj_to_tensor(adj: dict, num_entities: int, max_neighbors: int, device: str):
    """
    Convert adjacency dict to padded tensors for batch graph ops.

    Returns:
        neighbor_ids    : LongTensor [num_entities, max_neighbors]  (0-padded)
        neighbor_mask   : BoolTensor [num_entities, max_neighbors]  True = valid
    """
    neighbor_ids  = np.zeros((num_entities, max_neighbors), dtype=np.int64)
    neighbor_mask = np.zeros((num_entities, max_neighbors), dtype=bool)

    for eid, nbrs in adj.items():
        if eid >= num_entities:
            continue
        for k, n in enumerate(nbrs[:max_neighbors]):
            neighbor_ids[eid, k]  = n
            neighbor_mask[eid, k] = True

    return (
        torch.tensor(neighbor_ids,  device=device),
        torch.tensor(neighbor_mask, device=device),
    )


# ── Evaluation Metrics ─────────────────────────────────────────────────────────
def dcg_score(relevance: list, k: int) -> float:
    relevance = np.array(relevance[:k], dtype=np.float32)
    if relevance.size == 0:
        return 0.0
    gains = relevance / np.log2(np.arange(2, relevance.size + 2))
    return float(gains.sum())


def ndcg_score(y_true: list, y_score: list, k: int) -> float:
    order   = np.argsort(y_score)[::-1]
    y_true  = np.array(y_true)
    ideal   = sorted(y_true, reverse=True)
    dcg     = dcg_score(y_true[order].tolist(), k)
    idcg    = dcg_score(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_score(y_true: list, y_score: list) -> float:
    order  = np.argsort(y_score)[::-1]
    y_true = np.array(y_true)[order]
    for rank, label in enumerate(y_true, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def compute_metrics(all_labels: list, all_scores: list):
    """
    all_labels : list of lists (one per impression)
    all_scores : list of lists (one per impression)
    Returns dict with AUC, MRR, nDCG@5, nDCG@10
    """
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    for labels, scores in zip(all_labels, all_scores):
        if len(set(labels)) < 2:   # skip impressions with only one class
            continue
        aucs.append(roc_auc_score(labels, scores))
        mrrs.append(mrr_score(labels, scores))
        ndcg5s.append(ndcg_score(labels, scores, 5))
        ndcg10s.append(ndcg_score(labels, scores, 10))

    return {
        "AUC":     round(np.mean(aucs)    * 100, 2),
        "MRR":     round(np.mean(mrrs)    * 100, 2),
        "nDCG@5":  round(np.mean(ndcg5s)  * 100, 2),
        "nDCG@10": round(np.mean(ndcg10s) * 100, 2),
    }
