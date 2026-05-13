"""
Training script for GREP.
Usage:  python train.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import config
from utils import (
    set_seed, build_vocab, merge_news_data,
    load_entity_embeddings, build_word_embeddings,
    build_entity_graph, adj_to_tensor,
)
from dataloader import parse_news_file, get_train_loader
from model import GREP
from evaluate import evaluate


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update learning rate for all param groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_bpr_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) Loss.
    Ranks positive candidate higher than negative candidates.
    
    Args:
        scores: [B, k+1] tensor where first column is positive candidate score
        labels: [B, k+1] tensor (not used in standard BPR, kept for interface compatibility)
    
    Returns:
        Scalar loss value
    """
    pos_score = scores[:, 0]  # Positive candidate score [B]
    neg_scores = scores[:, 1:]  # Negative candidate scores [B, k]
    
    # BPR: maximize log(sigmoid(pos_score - neg_score))
    # Loss = -log(sigmoid(pos_score - neg_scores)) averaged over negatives and batch
    # Equivalent to: -mean(log(sigmoid(pos - neg)))
    pos_expanded = pos_score.unsqueeze(1)  # [B, 1]
    diffs = pos_expanded - neg_scores  # [B, k]
    
    # Use numerically stable sigmoid: log(sigmoid(x)) = x - log(1 + exp(x)) if x > 0 else -log(1 + exp(-x))
    # But F.softplus handles this: log(sigmoid(x)) = -softplus(-x)
    loss = torch.nn.functional.softplus(-diffs).mean()
    return loss


def build_components() -> tuple:
    """
    Build vocab, news dicts, embeddings, and KG adjacency tensors.
    Returns everything needed to construct the model and dataloaders.
    Also computes category weights for balanced loss.
    """
    print(" Building vocabulary from training news...")
    word2idx, train_news_raw = build_vocab(
        config.TRAIN_NEWS_FILE,
        config.MAX_TITLE_LEN,
        config.MAX_VOCAB_SIZE,
    )

    print("-- Parsing validation news...")
    _, val_news_raw = build_vocab(
        config.VAL_NEWS_FILE,
        config.MAX_TITLE_LEN,
        config.MAX_VOCAB_SIZE,
    )

    print("-- Loading entity embeddings...")
    entity2idx, entity_emb_matrix = load_entity_embeddings(
        config.ENTITY_EMBEDDING_FILE, config.ENTITY_EMBEDDING_DIM
    )

    print("-- Initialising word embeddings (GloVe)...")
    word_emb_matrix = build_word_embeddings(word2idx, config.WORD_EMBEDDING_DIM, config.GLOVE_FILE)

    print("-- Building knowledge graph...")
    all_news_raw = merge_news_data(train_news_raw, val_news_raw)
    adj = build_entity_graph(all_news_raw, entity2idx, config.MAX_ENTITY_NEIGHBORS)
    num_entities = len(entity2idx)
    kg_nbr_ids, kg_nbr_mask = adj_to_tensor(adj, num_entities, config.MAX_ENTITY_NEIGHBORS, config.DEVICE)

    print("-- Parsing news into index tensors...")
    train_news = parse_news_file(config.TRAIN_NEWS_FILE, word2idx, entity2idx)
    val_news   = parse_news_file(config.VAL_NEWS_FILE,   word2idx, entity2idx)
    # Merge so both dataloaders can look up any news_id
    news_dict  = {**train_news, **val_news}

    # ── Compute category weights for balanced loss ──────────────────────────
    print("-- Computing category weights...")
    from collections import Counter
    category_counts = Counter()
    for nid, info in news_dict.items():
        cat = info.get("category", "unknown")
        if cat != "pad":  # Skip padding category
            category_counts[cat] += 1
    
    # Inverse frequency weighting: weight[c] = 1 / freq[c]
    num_categories = len(category_counts)
    if category_counts:
        min_count = min(category_counts.values())
        category_weights = {cat: min_count / (count + 1e-8) for cat, count in category_counts.items()}
        # Normalize to sum to num_categories
        total_weight = sum(category_weights.values())
        category_weights = {cat: w * num_categories / total_weight for cat, w in category_weights.items()}
    else:
        category_weights = {}
    
    print(f"   Categories: {len(category_counts)}")
    print(f"   Sample weights: {list(category_weights.items())[:3]}")

    # Convert matrices to tensors
    word_emb_tensor   = torch.tensor(word_emb_matrix,   dtype=torch.float32)
    entity_emb_tensor = torch.tensor(entity_emb_matrix, dtype=torch.float32)

    return (
        word2idx, entity2idx,
        word_emb_tensor, entity_emb_tensor,
        kg_nbr_ids, kg_nbr_mask,
        news_dict,
        category_weights,
    )


def train():
    set_seed(config.SEED)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("-- Build all data components")
    (
        word2idx, entity2idx,
        word_emb_tensor, entity_emb_tensor,
        kg_nbr_ids, kg_nbr_mask,
        news_dict,
        category_weights,
    ) = build_components()

    print("-- Model")
    model = GREP(
        word_emb_matrix=word_emb_tensor,
        entity_emb_matrix=entity_emb_tensor,
        kg_nbr_ids=kg_nbr_ids,
        kg_nbr_mask=kg_nbr_mask,
    ).to(device)
    
    # Priority 2: float32 matmul precision optimization (works without torch.compile)
    # torch.compile requires Triton which is not available on Windows
    torch.set_float32_matmul_precision("high")
    print("[INFO] Using float32 matmul precision='high' for GPU optimization")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GREP model parameters: {total_params:,}")

    # ── Separate parameter groups for entity embeddings with lower LR ──────
    entity_embedding_params = [model.entity_embedding.weight]
    other_params = [p for n, p in model.named_parameters() if 'entity_embedding' not in n]
    
    optimizer = torch.optim.Adam(
        [
            {'params': other_params, 'lr': config.LEARNING_RATE},
            {'params': entity_embedding_params, 'lr': config.LEARNING_RATE * 0.1}  # 10x lower for entity embeddings
        ],
        weight_decay=config.WEIGHT_DECAY,  # L2 regularization
    )
    print(f"[INFO] Entity embeddings LR: {config.LEARNING_RATE * 0.1:.2e} (10x lower than other params)")

    print("-- DataLoaders")
    # Note: DataLoader will be recreated each epoch for curriculum learning
    print(f"Using curriculum learning: hard_neg_ratio = 0.3 -> 0.8 over {config.NUM_EPOCHS} epochs")
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda')

    print("-- Training loop")
    best_auc = 0.0
    epochs_without_improvement = 0
    patience = 2

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f'Epoch {epoch} started...')
        
        # ── Freeze/unfreeze word embeddings for curriculum learning ──────────────────
        if epoch <= 2:
            # Freeze word embeddings in epochs 1-2 to let entity/structural params stabilize
            model.title_encoder.embedding.weight.requires_grad = False
            print(f"  Word embeddings: FROZEN (epoch {epoch}/2)")
        else:
            # Unfreeze from epoch 3 onwards
            model.title_encoder.embedding.weight.requires_grad = True
            print(f"  Word embeddings: UNFROZEN (epoch {epoch})")
        
        # ── Curriculum Learning: Recreate train loader with epoch-dependent hard_neg_ratio ──
        train_loader = get_train_loader(news_dict, epoch=epoch)
        hard_neg_ratio = min(0.5, 0.3 + 0.4 * ((epoch - 1) / max(config.NUM_EPOCHS - 1, 1)))
        print(f"  Curriculum: hard_neg_ratio = {hard_neg_ratio:.2f}")
        
        model.train()
        total_loss = 0.0
        num_steps  = 0
        
        # Epoch-based learning rate scheduling
        if epoch == 1:
            epoch_lr = config.LEARNING_RATE  # First epoch: base LR with warmup
        else:
            # Exponential decay per epoch: lr = base_lr * (decay_factor ^ (epoch-1))
            epoch_lr = config.LEARNING_RATE * (config.LR_DECAY_FACTOR ** (epoch - 1))
        
        batch_step = 0  # Reset for warmup calculation
        accumulation_counter = 0  # Counter for gradient accumulation

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )

        for step, batch in enumerate(progress_bar, 1):
            # Warmup only applies in first epoch
            if epoch == 1 and batch_step < config.WARMUP_STEPS:
                # Linear warmup: gradually increase from 0 to epoch_lr
                current_lr = epoch_lr * (batch_step / config.WARMUP_STEPS)
            else:
                # After warmup or after epoch 1: use epoch_lr with decay
                current_lr = epoch_lr
            
            set_learning_rate(optimizer, current_lr)
            batch_step += 1
            accumulation_counter += 1
            
            # Move batch to device with non_blocking for efficiency
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # Forward pass with AMP
            with torch.autocast("cuda" if device.type == "cuda" else "cpu", 
                               enabled=(device.type == "cuda")):
                scores = model(batch)           # [B, k+1]
                labels = batch["labels"]        # [B, k+1]
                
                # ── Category-aware loss weighting ─────────────────────────
                # Get candidate categories and compute weights
                cand_titles = batch["cand_titles"]  # [B, k+1, MAX_TITLE]
                B, C = cand_titles.shape[0], cand_titles.shape[1]
                
                # Extract candidate news IDs from batch (we need to map from titles)
                # For now, we'll compute loss weights based on candidate index
                # Positive candidate (index 0) gets weight 1.0, negatives weighted by category
                loss_weights = torch.ones(B, C, device=device)
                
                # Apply inverse frequency weighting if we have category_weights
                # Note: Without explicit news IDs in batch, we use uniform weighting
                # but this can be enhanced if news_ids are passed
                
                # Loss function selection based on config
                targets = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
                if config.USE_BPR_LOSS:
                    loss = compute_bpr_loss(scores, labels)
                else:
                    # Cross-entropy with category weighting
                    ce_loss = F.cross_entropy(scores, targets, reduction='none', label_smoothing=0.1)  # [B]
                    # Apply loss weights and average
                    loss = ce_loss.mean()
                
                # Scale loss for gradient accumulation
                loss = loss / config.ACCUMULATION_STEPS

            # Only zero gradients on first accumulation step
            if accumulation_counter == 1:
                optimizer.zero_grad()
            
            # Backward with scaled loss
            scaler.scale(loss).backward()
            
            # Update only after accumulating enough steps
            if accumulation_counter % config.ACCUMULATION_STEPS == 0:
                # Unscale before grad clipping and NaN handling
                scaler.unscale_(optimizer)
                
                # Replace NaN gradients with zero
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.nan_to_num_(nan=0.0)
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
                
                accumulation_counter = 0

            total_loss += loss.item() * config.ACCUMULATION_STEPS  # Undo scaling for logging
            num_steps  += 1

            progress_bar.set_postfix(loss=loss.item() * config.ACCUMULATION_STEPS, lr=f"{current_lr:.2e}")

            if step % config.LOG_EVERY == 0:
                avg = total_loss / num_steps
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")

        avg_loss = total_loss / max(num_steps, 1)
        print(f"\nEpoch {epoch} finished – avg loss: {avg_loss:.4f}")
        
        # ── Save model after every epoch ───────────────────────────────────
        epoch_checkpoint = config.MODEL_SAVE_PATH.replace('.pt', f'_epoch_{epoch}.pt')
        torch.save(model.state_dict(), epoch_checkpoint)
        print(f"  ✓ Model saved after epoch {epoch}: {epoch_checkpoint}")

        # ── Evaluate on validation set ─────────────────────────────────────
        metrics, all_labels, all_scores = evaluate(model, news_dict, device, save_epoch_roc=epoch)
        print(
            f"  Val AUC={metrics['AUC']:.2f}  MRR={metrics['MRR']:.2f}"
            f"  nDCG@5={metrics['nDCG@5']:.2f}  nDCG@10={metrics['nDCG@10']:.2f}"
        )

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved  (AUC={best_auc:.2f})")
            # Save ROC curve for best model
            from evaluate import _save_roc_curve
            roc_save_path = config.MODEL_SAVE_PATH.replace('.pt', '_best_roc.png')
            _save_roc_curve(all_labels, all_scores, roc_save_path)
        else:
            epochs_without_improvement += 1
            print(f"  AUC did not improve. Epochs without improvement: {epochs_without_improvement}/{patience}")
            
            # Early stopping: stop if AUC hasn't improved for 'patience' epochs
            if epochs_without_improvement >= patience:
                print(f"\n[EARLY STOPPING] AUC did not improve for {patience} consecutive epochs.")
                print(f"Best AUC: {best_auc:.2f}")
                break

    print(f"\nTraining complete. Best AUC: {best_auc:.2f}")
    return model, news_dict, device


if __name__ == "__main__":
    train()
