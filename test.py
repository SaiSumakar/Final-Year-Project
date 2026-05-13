"""
Test script for GREP model on MIND test set.
Runs after training and evaluation to evaluate on held-out test data.

Usage:
    python test.py --checkpoint checkpoints/latest_2/grep_best.pt
    python test.py --checkpoint checkpoints/latest_2/grep_best.pt \\
                   --test-news-dir ./data/MINDlarge_test/news.tsv \\
                   --test-behaviors-dir ./data/MINDlarge_test/behaviors.tsv
    python test.py  # Uses defaults from config
"""

import argparse
import os
import json
from datetime import datetime
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from config import config
from utils import (
    set_seed, build_vocab, merge_news_data,
    load_entity_embeddings, build_word_embeddings,
    build_entity_graph, adj_to_tensor, compute_metrics,
)
from dataloader import parse_news_file, get_val_loader, MINDValDataset, DataLoader, collate_val
from model import GREP


# ── Test-specific dataloader (uses same validation logic as val) ─────────────
def get_test_loader(test_behaviors_file: str, news_dict: dict):
    """
    Create test dataloader. Reuses validation dataset logic since
    test set has same format as validation (variable candidates with labels).
    """
    dataset = MINDValDataset(test_behaviors_file, news_dict)
    return DataLoader(
        dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_val,
    )


@torch.no_grad()
def test(model: GREP, news_dict: dict, test_behaviors_file: str, device) -> tuple:
    """
    Run model on test behaviors with batched forward pass and AMP.
    
    Args:
        model: Trained GREP model
        news_dict: Dictionary of news articles
        test_behaviors_file: Path to test behaviors file
        device: Torch device
        
    Returns:
        (metrics_dict, all_labels, all_scores)
    """
    model.eval()
    test_loader = get_test_loader(test_behaviors_file, news_dict)

    all_labels, all_scores = [], []

    for batch in tqdm(test_loader, desc="Testing", leave=False):
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


def _save_test_roc_curve(all_labels, all_scores, save_path: str):
    """
    Generate and save ROC curve from test labels and scores.
    """
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
    plt.title('ROC Curve - GREP Model (Test Set)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _format_metrics_report(metrics: dict, stdout: bool = True) -> str:
    """
    Format metrics into a readable report string.
    
    Args:
        metrics: Dictionary with metrics
        stdout: If True, also print to stdout
        
    Returns:
        Formatted string suitable for file logging
    """
    report = "\n" + "=" * 50
    report += "\n         GREP Test Set Results"
    report += "\n" + "=" * 50
    report += f"\n  ROC-AUC  : {metrics['AUC']:.2f}%"
    report += f"\n  MRR      : {metrics['MRR']:.2f}%"
    report += f"\n  nDCG@5   : {metrics['nDCG@5']:.2f}%"
    report += f"\n  nDCG@10  : {metrics['nDCG@10']:.2f}%"
    report += "\n" + "=" * 50 + "\n"
    
    if stdout:
        print(report)
    
    return report


def _save_metrics_to_file(metrics: dict, model_path: str, log_dir: str = "outputs"):
    """
    Save metrics to a JSON and text file for verification.
    
    Args:
        metrics: Dictionary with metrics
        model_path: Path to the model checkpoint used
        log_dir: Directory to save logs
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_results_{timestamp}.txt")
    json_file = os.path.join(log_dir, f"test_metrics_{timestamp}.json")
    
    # Save as text format
    with open(log_file, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("GREP Model - Test Set Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint: {model_path}\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"ROC-AUC  : {metrics['AUC']:.2f}%\n")
        f.write(f"MRR      : {metrics['MRR']:.2f}%\n")
        f.write(f"nDCG@5   : {metrics['nDCG@5']:.2f}%\n")
        f.write(f"nDCG@10  : {metrics['nDCG@10']:.2f}%\n")
        f.write("-" * 50 + "\n")
    
    # Save as JSON format
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model_checkpoint": model_path,
            "metrics": metrics
        }, f, indent=2)
    
    print(f"\n✓ Test results saved:")
    print(f"  - Text report: {log_file}")
    print(f"  - JSON metrics: {json_file}")
    
    return log_file, json_file


def run_test(
    checkpoint_path: str = None,
    test_news_file: str = None,
    test_behaviors_file: str = None,
    train_news_file: str = None,
    val_news_file: str = None,
    entity_embedding_file: str = None,
    device: str = None,
):
    """
    Full test pipeline: load model, load data, run inference, save results.
    
    Args:
        checkpoint_path: Path to best model checkpoint (default: config.MODEL_SAVE_PATH)
        test_news_file: Path to test news.tsv (default: config.TEST_NEWS_FILE)
        test_behaviors_file: Path to test behaviors.tsv (default: config.TEST_BEHAVIORS_FILE)
        train_news_file: Path to training news.tsv for vocab (default: config.TRAIN_NEWS_FILE)
        val_news_file: Path to validation news.tsv (default: config.VAL_NEWS_FILE)
        entity_embedding_file: Path to entity embeddings (default: config.ENTITY_EMBEDDING_FILE)
        device: Device to use (default: config.DEVICE)
    """
    # Use config defaults if paths not provided
    if checkpoint_path is None:
        checkpoint_path = config.MODEL_SAVE_PATH
    if test_news_file is None:
        test_news_file = config.TEST_NEWS_FILE
    if test_behaviors_file is None:
        test_behaviors_file = config.TEST_BEHAVIORS_FILE
    if train_news_file is None:
        train_news_file = config.TRAIN_NEWS_FILE
    if val_news_file is None:
        val_news_file = config.VAL_NEWS_FILE
    if entity_embedding_file is None:
        entity_embedding_file = config.ENTITY_EMBEDDING_FILE
    if device is None:
        device = config.DEVICE
    
    set_seed(config.SEED)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Verify checkpoint exists ────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  Please ensure training is complete and model is saved.")
        return False

    # ── Build components ───────────────────────────────────────────────────
    print("── Building vocabulary from training news…")
    word2idx, train_news_raw = build_vocab(
        train_news_file, config.MAX_TITLE_LEN, config.MAX_VOCAB_SIZE
    )

    print("-- Parsing validation news (for vocab expansion)…")
    _, val_news_raw = build_vocab(
        val_news_file, config.MAX_TITLE_LEN, config.MAX_VOCAB_SIZE
    )

    print("-- Loading entity embeddings…")
    entity2idx, entity_emb_matrix = load_entity_embeddings(
        entity_embedding_file, config.ENTITY_EMBEDDING_DIM
    )

    print("-- Initialising word embeddings…")
    word_emb_matrix = build_word_embeddings(word2idx, config.WORD_EMBEDDING_DIM, config.GLOVE_FILE)

    print("-- Building knowledge graph from train+val+test news…")
    # Try to parse test news for KG building
    test_news_for_kg = {}
    if os.path.exists(test_news_file):
        from dataloader import parse_news_file as parse_for_kg_building
        try:
            test_news_raw = build_vocab(
                test_news_file, config.MAX_TITLE_LEN, config.MAX_VOCAB_SIZE
            )[1]
        except Exception as e:
            print(f"  Warning: Could not parse test news for KG: {e}")
            test_news_raw = {}
    else:
        test_news_raw = {}
    
    all_news_raw = merge_news_data(train_news_raw, val_news_raw)
    all_news_raw = merge_news_data(all_news_raw, test_news_raw)
    
    adj = build_entity_graph(all_news_raw, entity2idx, config.MAX_ENTITY_NEIGHBORS)
    num_entities = len(entity2idx)
    kg_nbr_ids, kg_nbr_mask = adj_to_tensor(
        adj, num_entities, config.MAX_ENTITY_NEIGHBORS, str(device)
    )

    print("-- Parsing news files into index tensors…")
    train_news = parse_news_file(train_news_file, word2idx, entity2idx)
    val_news   = parse_news_file(val_news_file, word2idx, entity2idx)
    test_news  = parse_news_file(test_news_file, word2idx, entity2idx)
    # Merge all so any dataloader can look up any news_id
    news_dict  = {**train_news, **val_news, **test_news}

    # ── Build model ────────────────────────────────────────────────────────
    print("-- Building GREP model…")
    word_emb_tensor   = torch.tensor(word_emb_matrix,   dtype=torch.float32)
    entity_emb_tensor = torch.tensor(entity_emb_matrix, dtype=torch.float32)

    model = GREP(
        word_emb_matrix=word_emb_tensor,
        entity_emb_matrix=entity_emb_tensor,
        kg_nbr_ids=kg_nbr_ids,
        kg_nbr_mask=kg_nbr_mask,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GREP model parameters: {total_params:,}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"\n-- Loading checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False

    # ── Run test ───────────────────────────────────────────────────────────
    print(f"\n-- Running inference on test set…")
    if not os.path.exists(test_behaviors_file):
        print(f"✗ Test behaviors file not found: {test_behaviors_file}")
        return False

    metrics, all_labels, all_scores = test(model, news_dict, test_behaviors_file, device)

    # ── Save results ───────────────────────────────────────────────────────
    print("\n-- Saving results…")
    text_file, json_file = _save_metrics_to_file(metrics, checkpoint_path)

    # Save ROC curve
    roc_save_path = os.path.join(os.path.dirname(text_file), "test_roc_curve.png")
    _save_test_roc_curve(all_labels, all_scores, roc_save_path)
    print(f"  ✓ ROC curve saved: {roc_save_path}")

    # Print metrics to console
    _format_metrics_report(metrics, stdout=True)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test GREP model on MIND test set"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=config.MODEL_SAVE_PATH,
        help=f"Path to saved model checkpoint (default: {config.MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        "--test-news-file", type=str, default=config.TEST_NEWS_FILE,
        help=f"Path to test news.tsv file (default: {config.TEST_NEWS_FILE})"
    )
    parser.add_argument(
        "--test-behaviors-file", type=str, default=config.TEST_BEHAVIORS_FILE,
        help=f"Path to test behaviors.tsv file (default: {config.TEST_BEHAVIORS_FILE})"
    )
    parser.add_argument(
        "--train-news-file", type=str, default=config.TRAIN_NEWS_FILE,
        help="Path to train news.tsv file (for vocab)"
    )
    parser.add_argument(
        "--val-news-file", type=str, default=config.VAL_NEWS_FILE,
        help="Path to val news.tsv file"
    )
    parser.add_argument(
        "--entity-embedding-file", type=str, default=config.ENTITY_EMBEDDING_FILE,
        help="Path to entity embeddings"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()

    success = run_test(
        checkpoint_path=args.checkpoint,
        test_news_file=args.test_news_file,
        test_behaviors_file=args.test_behaviors_file,
        train_news_file=args.train_news_file,
        val_news_file=args.val_news_file,
        entity_embedding_file=args.entity_embedding_file,
        device=args.device,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
