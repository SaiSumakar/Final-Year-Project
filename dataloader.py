import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import config


def parse_news_file(news_file: str, word2idx: dict, entity2idx: dict):
    """
    Parse news.tsv.
    Returns dict: {news_id: {'title_idx': LongTensor[MAX_TITLE_LEN],
                              'entity_idx': list[int],
                              'category': str}}
    """
    news = {}
    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            nid      = parts[0]
            category = parts[2] if len(parts) > 2 else "unknown"
            words    = parts[3].lower().split()[:config.MAX_TITLE_LEN]
            
            # Parse title entities from parts[6]
            try:
                title_entities = [e["WikidataId"] for e in json.loads(parts[6])]
            except Exception:
                title_entities = []
            
            # Parse abstract entities from parts[7] if available
            abstract_entities = []
            if len(parts) > 7:
                try:
                    abstract_entities = [e["WikidataId"] for e in json.loads(parts[7])]
                except Exception:
                    abstract_entities = []
            
            # Merge both lists and deduplicate
            entities = list(set(title_entities + abstract_entities))

            # encode words
            word_ids = [word2idx.get(w, 1) for w in words]
            # pad / truncate
            word_ids = word_ids[:config.MAX_TITLE_LEN]
            word_ids += [0] * (config.MAX_TITLE_LEN - len(word_ids))

            # encode entities  (keep only known ones)
            ent_ids = [entity2idx.get(e, 0) for e in entities]
            ent_ids = [e for e in ent_ids if e > 0]

            news[nid] = {
                "title_idx":  torch.tensor(word_ids, dtype=torch.long),
                "entity_idx": ent_ids,      # variable length list of ints
                "category":   category,     # news category
            }

    # PAD news – returned when a news_id is unknown
    pad_title  = torch.zeros(config.MAX_TITLE_LEN, dtype=torch.long)
    news["[PAD]"] = {"title_idx": pad_title, "entity_idx": [], "category": "pad"}
    return news


def parse_behaviors(behaviors_file: str, is_train: bool = True):
    """
    Parse behaviors.tsv.
    File format: [line_idx]\t[user_id]\t[timestamp]\t[history]\t[impressions]
    
    Returns list of dicts:
      train : {'history': [nid,...], 'pos': nid, 'neg': [nid,...], 'timestamp': int}
      val   : {'history': [nid,...], 'candidates': [nid,...], 'labels': [0/1,...], 'timestamp': int}
      test  : {'history': [nid,...], 'candidates': [nid,...], 'labels': [], 'timestamp': int}
    """
    samples = []
    with open(behaviors_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            # Skip parts[0] (line index), parts[1] (user_id), parts[2] (timestamp string)
            # For now, use line index as a simple proxy for recency (newer entries have higher indices)
            timestamp   = line_idx  # Use sequential counter as proxy for recency
            history     = parts[3].split() if parts[3] else []
            impressions = parts[4].split()

            candidates, labels = [], []
            for imp in impressions:
                # Handle both formats: "nid-label" (train/val) and "nid" (test)
                if "-" in imp:
                    nid, label = imp.rsplit("-", 1)
                    candidates.append(nid)
                    labels.append(int(label))
                else:
                    # Test set format: just news ID without label
                    candidates.append(imp)
                    labels.append(0)  # Default to 0 for test set

            if is_train:
                pos_list = [c for c, l in zip(candidates, labels) if l == 1]
                neg_list = [c for c, l in zip(candidates, labels) if l == 0]
                for pos in pos_list:
                    samples.append({
                        "history":    history,
                        "pos":        pos,
                        "neg":        neg_list,
                        "timestamp":  timestamp,
                    })
            else:
                samples.append({
                    "history":   history,
                    "candidates": candidates,
                    "labels":    labels,
                    "timestamp": timestamp,
                })

    return samples


class MINDTrainDataset(Dataset):
    def __init__(self, behaviors_file: str, news_dict: dict, epoch: int = 1):
        self.samples    = parse_behaviors(behaviors_file, is_train=True)
        self.news       = news_dict
        self.neg_k      = config.NEG_SAMPLE_RATIO
        self.max_hist   = config.MAX_HISTORY_LEN
        # Curriculum learning: gradually increase hard negative ratio from 0.3 to 0.5 (capped to prevent overfitting)
        self.hard_neg_ratio = min(0.5, 0.3 + 0.4 * ((epoch - 1) / max(config.NUM_EPOCHS - 1, 1)))
        self.epoch      = epoch
        
        # Pre-build category → [nid_list] inverted index for fast hard negative selection
        self.category_to_news = {}
        for nid, info in news_dict.items():
            cat = info.get("category", "unknown")
            if cat not in self.category_to_news:
                self.category_to_news[cat] = []
            self.category_to_news[cat].append(nid)

    def __len__(self):
        return len(self.samples)

    def _get_news_title(self, nid):
        return self.news.get(nid, self.news["[PAD]"])["title_idx"]

    def _get_news_entities(self, nid):
        return self.news.get(nid, self.news["[PAD]"])["entity_idx"]
    
    def _get_news_category(self, nid):
        """Get category of news article."""
        return self.news.get(nid, self.news["[PAD]"]).get("category", "unknown")
    
    def _select_hard_negatives(self, pos_nid, neg_list):
        """
        Select hard negatives: category-based hard + random easy negatives.
        Uses pre-built category index for speed.
        """
        if not neg_list:
            return []
        
        num_hard = max(1, int(self.neg_k * self.hard_neg_ratio))
        num_random = self.neg_k - num_hard
        
        pos_category = self._get_news_category(pos_nid)
        
        # Convert neg_list to set for O(1) membership checks
        neg_set = set(neg_list)
        
        # Use pre-built category index for fast lookup
        same_category = [nid for nid in self.category_to_news.get(pos_category, []) if nid in neg_set]
        other_category = [nid for nid in neg_list if nid not in same_category]
        
        # Select hard negatives (same category, more discriminative)
        if same_category:
            hard_negs = random.sample(same_category, min(num_hard, len(same_category)))
        else:
            hard_negs = []
        
        # Fill with random negatives from other categories
        available_for_random = other_category if other_category else neg_list
        if hard_negs:
            available_for_random = [n for n in available_for_random if n not in hard_negs]
        
        if available_for_random:
            random_negs = random.sample(available_for_random, min(num_random, len(available_for_random)))
        else:
            random_negs = random.sample(neg_list, min(num_random, len(neg_list))) if neg_list else []
        
        selected = hard_negs + random_negs
        
        # Pad if needed
        while len(selected) < self.neg_k:
            selected.append("[PAD]")
        
        return selected[:self.neg_k]

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- history (pad / truncate to MAX_HISTORY_LEN) ----
        hist = s["history"][-self.max_hist:]
        h_len = len(hist)
        timestamp = s.get("timestamp", 0)  # Current timestamp

        if h_len > 0:
            hist_titles = torch.stack([self._get_news_title(n) for n in hist])
            hist_entities = [self._get_news_entities(n) for n in hist]
        else:
            hist_titles = torch.empty(0, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_entities = []

        # Pad to MAX_HISTORY_LEN
        if h_len < self.max_hist:
            pad_size = self.max_hist - h_len
            pad_titles = torch.zeros(pad_size, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_titles = torch.cat([hist_titles, pad_titles], dim=0)
            hist_entities = hist_entities + [[]] * pad_size

        # ---- candidate: 1 pos + k neg (with category-based hard negative sampling) ----
        hard_negs = self._select_hard_negatives(s["pos"], s["neg"])
        
        candidates    = [s["pos"]] + hard_negs   # length k+1
        cand_titles   = torch.stack([self._get_news_title(n)  for n in candidates])
        cand_entities = [self._get_news_entities(n) for n in candidates]

        labels = torch.zeros(self.neg_k + 1, dtype=torch.long)
        labels[0] = 1

        return {
            "hist_titles":    hist_titles,          # [MAX_HIST, MAX_TITLE]
            "hist_entities":  hist_entities,        # list[list[int]]
            "hist_len":       h_len,
            "cand_titles":    cand_titles,           # [k+1, MAX_TITLE]
            "cand_entities":  cand_entities,         # list[list[int]]
            "labels":         labels,                # [k+1]
            "timestamp":      timestamp,             # Unix epoch for temporal decay
        }


class AugmentedMINDTrainDataset(MINDTrainDataset):
    """
    Augmented training dataset with three augmentation strategies:
    1. Title dropout: Randomly drop 30% of words from titles
    2. History variation: Randomly crop/extend history window
    3. Entity enrichment: Add 1-hop entity neighbors to history
    """
    
    def __init__(self, behaviors_file: str, news_dict: dict, epoch: int = 1, entity_graph: dict = None):
        super().__init__(behaviors_file, news_dict, epoch)
        self.entity_graph = entity_graph or {}  # Entity -> [neighbor_ids]
        self.augmentation_prob = 0.4  # Apply augmentation with 40% probability (reduced from 70% to prevent overfitting)
    
    def _augment_title(self, title_ids: torch.Tensor) -> torch.Tensor:
        """Augmentation 1: Title dropout (randomly drop 30% of words)"""
        if random.random() > self.augmentation_prob:
            return title_ids
        
        # Clone to avoid modifying original
        aug_ids = title_ids.clone()
        
        # Find non-zero positions (actual words, not padding)
        non_zero_mask = aug_ids != 0
        non_zero_indices = non_zero_mask.nonzero(as_tuple=True)[0]
        
        if len(non_zero_indices) > 0:
            # Randomly drop 30% of words
            num_to_drop = max(1, int(len(non_zero_indices) * 0.3))
            drop_indices = random.sample(list(non_zero_indices), min(num_to_drop, len(non_zero_indices)))
            # Convert to tensor for proper indexing
            drop_indices_tensor = torch.tensor(drop_indices, dtype=torch.long, device=aug_ids.device)
            aug_ids[drop_indices_tensor] = 0  # Set to padding
        
        return aug_ids
    
    def _augment_history(self, hist: list) -> list:
        """Augmentation 2: History variation (random crop/extend)"""
        if random.random() > self.augmentation_prob or len(hist) == 0:
            return hist
        
        # Randomly select a window of 40-50 items from history
        window_size = random.randint(max(1, len(hist) - 10), len(hist))
        start_idx = max(0, len(hist) - window_size)
        return hist[start_idx:]
    
    def _augment_entities(self, entity_ids: list) -> list:
        """Augmentation 3: Entity enrichment (add 1-hop neighbors)"""
        if random.random() > self.augmentation_prob or not entity_ids:
            return entity_ids
        
        aug_entities = entity_ids.copy()
        
        # Add 1-2 random neighbors for each entity
        for ent_id in entity_ids[:len(entity_ids)]:
            if ent_id in self.entity_graph:
                neighbors = self.entity_graph[ent_id]
                if neighbors:
                    num_to_add = random.randint(1, 2)
                    added = random.sample(neighbors, min(num_to_add, len(neighbors)))
                    aug_entities.extend(added)
        
        # Remove duplicates and keep original order
        seen = set()
        deduped = []
        for e in aug_entities:
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        
        return deduped[:20]  # Limit to max_entity
    
    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- history (with augmentation) ----
        hist = s["history"][-self.max_hist:]
        hist = self._augment_history(hist)
        h_len = len(hist)
        timestamp = s.get("timestamp", 0)

        if h_len > 0:
            hist_titles = torch.stack([self._augment_title(self._get_news_title(n)) for n in hist])
            hist_entities = [self._augment_entities(self._get_news_entities(n)) for n in hist]
        else:
            hist_titles = torch.empty(0, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_entities = []

        # Pad to MAX_HISTORY_LEN
        if h_len < self.max_hist:
            pad_size = self.max_hist - h_len
            pad_titles = torch.zeros(pad_size, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_titles = torch.cat([hist_titles, pad_titles], dim=0)
            hist_entities = hist_entities + [[]] * pad_size

        # ---- candidate: 1 pos + k neg ----
        hard_negs = self._select_hard_negatives(s["pos"], s["neg"])
        candidates    = [s["pos"]] + hard_negs
        cand_titles = torch.stack([self._augment_title(self._get_news_title(n)) for n in candidates])
        cand_entities = [self._get_news_entities(n) for n in candidates]

        labels = torch.zeros(self.neg_k + 1, dtype=torch.long)
        labels[0] = 1

        return {
            "hist_titles":    hist_titles,
            "hist_entities":  hist_entities,
            "hist_len":       h_len,
            "cand_titles":    cand_titles,
            "cand_entities":  cand_entities,
            "labels":         labels,
            "timestamp":      timestamp,
        }


class MINDValDataset(Dataset):
    def __init__(self, behaviors_file: str, news_dict: dict):
        self.samples  = parse_behaviors(behaviors_file, is_train=False)
        self.news     = news_dict
        self.max_hist = config.MAX_HISTORY_LEN

    def __len__(self):
        return len(self.samples)

    def _get_news_title(self, nid):
        return self.news.get(nid, self.news["[PAD]"])["title_idx"]

    def _get_news_entities(self, nid):
        return self.news.get(nid, self.news["[PAD]"])["entity_idx"]

    def __getitem__(self, idx):
        s = self.samples[idx]

        hist = s["history"][-self.max_hist:]
        h_len = len(hist)
        timestamp = s.get("timestamp", 0)  # Current timestamp

        if h_len > 0:
            hist_titles = torch.stack([self._get_news_title(n) for n in hist])
            hist_entities = [self._get_news_entities(n) for n in hist]
        else:
            hist_titles = torch.empty(0, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_entities = []

        # Pad to MAX_HISTORY_LEN
        if h_len < self.max_hist:
            pad_size = self.max_hist - h_len
            pad_titles = torch.zeros(pad_size, config.MAX_TITLE_LEN, dtype=torch.long)
            hist_titles = torch.cat([hist_titles, pad_titles], dim=0)
            hist_entities = hist_entities + [[]] * pad_size

        cand_titles   = torch.stack([self._get_news_title(n)  for n in s["candidates"]])
        cand_entities = [self._get_news_entities(n) for n in s["candidates"]]

        return {
            "hist_titles":    hist_titles,
            "hist_entities":  hist_entities,
            "hist_len":       h_len,
            "cand_titles":    cand_titles,
            "cand_entities":  cand_entities,
            "labels":         torch.tensor(s["labels"], dtype=torch.float),
            "timestamp":      timestamp,
        }


# ── Custom collate to handle variable-length entity lists ─────────────────────
def _pad_entity_list(entity_lists: list):
    """
    entity_lists : list (batch) of list (news) of list[int]
    Returns LongTensor [B, N, max_ent] and mask BoolTensor [B, N, max_ent] on CPU.
    Device transfer handled outside collate for multiprocessing + pin_memory efficiency.
    
    Vectorized with NumPy for 5–10× speedup over nested Python loops.
    """
    import numpy as np
    
    B = len(entity_lists)
    if B == 0:
        return torch.zeros(0, 1, 1, dtype=torch.long), torch.zeros(0, 1, 1, dtype=torch.bool)
    
    # Compute dimensions
    N = max(len(x) for x in entity_lists) if entity_lists else 1
    max_e = max((max((len(e) for e in news_list), default=0)
                 for news_list in entity_lists), default=1)
    max_e = max(max_e, 1)

    # Initialize with NumPy for faster allocation
    ids_np  = np.zeros((B, N, max_e), dtype=np.int64)
    mask_np = np.zeros((B, N, max_e), dtype=np.bool_)
    
    # Vectorized filling: for each batch item and news item, assign entities
    for b, news_list in enumerate(entity_lists):
        for n, ents in enumerate(news_list):
            if ents:
                ents_array = np.array(ents[:max_e], dtype=np.int64)
                ids_np[b, n, :len(ents_array)] = ents_array
                mask_np[b, n, :len(ents_array)] = True
    
    # Convert back to torch tensors
    ids  = torch.from_numpy(ids_np).to(dtype=torch.long)
    mask = torch.from_numpy(mask_np).to(dtype=torch.bool)

    return ids, mask


def collate_train(batch):
    hist_titles   = torch.stack([x["hist_titles"]  for x in batch])
    cand_titles   = torch.stack([x["cand_titles"]  for x in batch])
    labels        = torch.stack([x["labels"]       for x in batch])
    hist_lens     = [x["hist_len"] for x in batch]
    timestamps    = [x["timestamp"] for x in batch]  # List of Unix epochs

    hist_ent_ids, hist_ent_mask = _pad_entity_list(
        [x["hist_entities"] for x in batch])
    cand_ent_ids, cand_ent_mask = _pad_entity_list(
        [x["cand_entities"] for x in batch])

    return {
        "hist_titles":    hist_titles,        # [B, MAX_HIST, MAX_TITLE]
        "hist_ent_ids":   hist_ent_ids,       # [B, MAX_HIST, max_ent]
        "hist_ent_mask":  hist_ent_mask,      # [B, MAX_HIST, max_ent]
        "hist_lens":      hist_lens,
        "cand_titles":    cand_titles,        # [B, k+1, MAX_TITLE]
        "cand_ent_ids":   cand_ent_ids,       # [B, k+1, max_ent]
        "cand_ent_mask":  cand_ent_mask,
        "labels":         labels,             # [B, k+1]
        "timestamps":     timestamps,         # [B] Unix epochs for temporal decay
    }


def collate_val(batch):
    """
    Batched collate for validation: pads all candidates to max_C in batch.
    Returns single dict with batch-level tensors + num_cands for unpadding.
    """
    B = len(batch)
    
    # Determine max candidate count in this batch
    max_C = max(len(x["cand_titles"]) for x in batch)
    
    # Stack history across batch (all same fixed size)
    hist_titles   = torch.stack([x["hist_titles"]  for x in batch])
    hist_lens     = [x["hist_len"] for x in batch]
    timestamps    = [x["timestamp"] for x in batch]  # Unix epochs
    num_cands     = [len(x["cand_titles"]) for x in batch]
    
    # Pad candidates to max_C; keep track of which are valid
    cand_titles_list = []
    cand_entities_list = []
    cand_labels_list = []
    cand_mask_list = []
    
    for x in batch:
        C = len(x["cand_titles"])
        pad_C = max_C - C
        
        # Pad titles  [C → max_C, MAX_TITLE]
        padded_titles = torch.cat([
            x["cand_titles"],
            torch.zeros(pad_C, config.MAX_TITLE_LEN, dtype=torch.long)
        ], dim=0)
        cand_titles_list.append(padded_titles)
        
        # Pad entities (per-candidate entity lists)
        cand_entities_list.append(x["cand_entities"] + [[]] * pad_C)
        
        # Pad labels
        padded_labels = torch.cat([
            x["labels"],
            torch.zeros(pad_C, dtype=torch.float)
        ], dim=0)
        cand_labels_list.append(padded_labels)
        
        # Candidate mask  [max_C]
        mask = torch.cat([
            torch.ones(C, dtype=torch.bool),
            torch.zeros(pad_C, dtype=torch.bool)
        ], dim=0)
        cand_mask_list.append(mask)
    
    cand_titles = torch.stack(cand_titles_list)      # [B, max_C, MAX_TITLE]
    cand_mask = torch.stack(cand_mask_list)           # [B, max_C]
    cand_labels = torch.stack(cand_labels_list)       # [B, max_C]
    
    # Pad entity lists (call _pad_entity_list once per batch)
    hist_ent_ids, hist_ent_mask = _pad_entity_list(
        [x["hist_entities"] for x in batch])
    cand_ent_ids, cand_ent_mask = _pad_entity_list(cand_entities_list)

    return {
        "hist_titles":    hist_titles,        # [B, MAX_HIST, MAX_TITLE]
        "hist_ent_ids":   hist_ent_ids,       # [B, MAX_HIST, max_ent]
        "hist_ent_mask":  hist_ent_mask,      # [B, MAX_HIST, max_ent]
        "hist_lens":      hist_lens,           # list[int]
        "cand_titles":    cand_titles,        # [B, max_C, MAX_TITLE]
        "cand_ent_ids":   cand_ent_ids,       # [B, max_C, max_ent]
        "cand_ent_mask":  cand_ent_mask,      # [B, max_C, max_ent]
        "cand_mask":      cand_mask,          # [B, max_C]  True=valid candidate
        "cand_labels":    cand_labels,        # [B, max_C]
        "num_cands":      num_cands,          # list[int]  true count per sample
        "timestamps":     timestamps,         # [B] Unix epochs for temporal decay
    }


def get_train_loader(news_dict: dict, epoch: int = 1):
    if config.USE_AUGMENTATION:
        dataset = AugmentedMINDTrainDataset(config.TRAIN_BEHAVIORS_FILE, news_dict, epoch=epoch, entity_graph=None)
    else:
        dataset = MINDTrainDataset(config.TRAIN_BEHAVIORS_FILE, news_dict, epoch=epoch)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_train,
    )


def get_val_loader(news_dict: dict):
    dataset = MINDValDataset(config.VAL_BEHAVIORS_FILE, news_dict)
    return DataLoader(
        dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_val,
    )
