"""
Evaluate a trained GREP model on the MIND validation set.
Prints: ROC-AUC, MRR, nDCG@5, nDCG@10

Can be run standalone:
    python evaluate.py --checkpoint checkpoints/grep_best.pt
or imported and called as evaluate(model, news_dict, device).
"""

import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

from config import config
from utils import (
    set_seed, build_vocab, merge_news_data,
    load_entity_embeddings, build_word_embeddings,
    build_entity_graph, adj_to_tensor,
    compute_metrics,
)
from dataloader import parse_news_file, get_val_loader
from model import GREP


@torch.no_grad()
def evaluate(model: GREP, news_dict: dict, device, save_epoch_roc: int = None) -> tuple:
    """
    Run model on validation behaviors with batched forward pass and AMP.
    """
    model.eval()
    val_loader = get_val_loader(news_dict)

    all_labels, all_scores = [], []

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        # Move batch to device with non_blocking
        hist_titles   = batch["hist_titles"].to(device, non_blocking=True)
        hist_ent_ids  = batch["hist_ent_ids"].to(device, non_blocking=True)
        hist_ent_mask = batch["hist_ent_mask"].to(device, non_blocking=True)
        cand_titles   = batch["cand_titles"].to(device, non_blocking=True)
        cand_ent_ids  = batch["cand_ent_ids"].to(device, non_blocking=True)
        cand_ent_mask = batch["cand_ent_mask"].to(device, non_blocking=True)
        cand_mask     = batch["cand_mask"].to(device, non_blocking=True)
        cand_labels   = batch["cand_labels"].to(device, non_blocking=True)
        num_cands     = batch["num_cands"]
        
        # Forward pass with AMP
        with torch.autocast("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            batch_data = {
                "hist_titles":   hist_titles,
                "hist_ent_ids":  hist_ent_ids,
                "hist_ent_mask": hist_ent_mask,
                "hist_lens":     batch["hist_lens"],
                "cand_titles":   cand_titles,
                "cand_ent_ids":  cand_ent_ids,
                "cand_ent_mask": cand_ent_mask,
            }
            scores = model(batch_data)  # [B, max_C]
        
        # Strip padding using num_cands
        B = scores.shape[0]
        for b in range(B):
            C = num_cands[b]
            score_b = scores[b, :C].cpu().tolist()
            label_b = cand_labels[b, :C].cpu().tolist()
            all_scores.append(score_b)
            all_labels.append(label_b)

    metrics = compute_metrics(all_labels, all_scores)
    return metrics, all_labels, all_scores


def _save_roc_curve(all_labels, all_scores, save_path="checkpoints/roc_curve.png"):
    """
    Generate and save ROC curve from labels and scores.
    """
    # Flatten labels and scores
    flat_labels = []
    flat_scores = []
    
    for labels, scores in zip(all_labels, all_scores):
        flat_labels.extend(labels)
        flat_scores.extend(scores)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - GREP Model')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved: {save_path}")


def _print_metrics(metrics: dict):
    print("\n" + "=" * 45)
    print("          GREP Evaluation Results")
    print("=" * 45)
    print(f"  ROC-AUC  : {metrics['AUC']:.2f}")
    print(f"  MRR      : {metrics['MRR']:.2f}")
    print(f"  nDCG@5   : {metrics['nDCG@5']:.2f}")
    print(f"  nDCG@10  : {metrics['nDCG@10']:.2f}")
    print("=" * 45 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=config.MODEL_SAVE_PATH,
        help="Path to saved model checkpoint (.pt)"
    )
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("── Building vocab …")
    word2idx, train_news_raw = build_vocab(
        config.TRAIN_NEWS_FILE, config.MAX_TITLE_LEN, config.MAX_VOCAB_SIZE
    )
    _, val_news_raw = build_vocab(
        config.VAL_NEWS_FILE, config.MAX_TITLE_LEN, config.MAX_VOCAB_SIZE
    )

    print("── Loading entity embeddings …")
    entity2idx, entity_emb_matrix = load_entity_embeddings(
        config.ENTITY_EMBEDDING_FILE, config.ENTITY_EMBEDDING_DIM
    )

    print("── Initialising word embeddings (random) …")
    word_emb_matrix = build_word_embeddings(word2idx, config.WORD_EMBEDDING_DIM)

    print("── Building knowledge graph …")
    all_news_raw = merge_news_data(train_news_raw, val_news_raw)
    adj = build_entity_graph(all_news_raw, entity2idx, config.MAX_ENTITY_NEIGHBORS)
    num_entities = len(entity2idx)
    kg_nbr_ids, kg_nbr_mask = adj_to_tensor(
        adj, num_entities, config.MAX_ENTITY_NEIGHBORS, str(device)
    )

    print("── Parsing news …")
    train_news = parse_news_file(config.TRAIN_NEWS_FILE, word2idx, entity2idx)
    val_news   = parse_news_file(config.VAL_NEWS_FILE,   word2idx, entity2idx)
    news_dict  = {**train_news, **val_news}

    word_emb_tensor   = torch.tensor(word_emb_matrix,   dtype=torch.float32)
    entity_emb_tensor = torch.tensor(entity_emb_matrix, dtype=torch.float32)

    model = GREP(
        word_emb_matrix=word_emb_tensor,
        entity_emb_matrix=entity_emb_tensor,
        kg_nbr_ids=kg_nbr_ids,
        kg_nbr_mask=kg_nbr_mask,
    ).to(device)

    print(f"── Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    print("── Running evaluation on validation set …")
    metrics, all_labels, all_scores = evaluate(model, news_dict, device)
    _print_metrics(metrics)
    _save_roc_curve(all_labels, all_scores)


if __name__ == "__main__":
    main()
