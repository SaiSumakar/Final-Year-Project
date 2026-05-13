"""
GREP: Graph neural news Recommendation with user Existing and Potential interest modeling
Paper: Qiu et al., ACM TKDD 2022  –  https://doi.org/10.1145/3511708
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


# ── Helpers ────────────────────────────────────────────────────────────────────

class GatedAggregation(nn.Module):
    """Equation (2) in the paper – soft-attention weighted sum over a sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.Wa = nn.Linear(dim, dim, bias=True)
        self.Wg = nn.Linear(dim, 1,   bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        """
        H    : [*, seq, dim]
        mask : [*, seq]  BoolTensor – True for valid positions
        Returns [*, dim]
        """
        scores = self.Wg(torch.tanh(self.Wa(H))).squeeze(-1)   # [*, seq]
        
        if mask is not None:
            # Check which rows are all-masked (no valid positions)
            all_masked = ~mask.any(dim=-1, keepdim=True)  # [*, 1]
            # Set scores to zero for masked positions, but avoid -inf to prevent NaN gradients
            scores = scores.masked_fill(~mask, float("-1e4"))  # Use large negative safe for fp16
        
        r = F.softmax(scores, dim=-1).unsqueeze(-1)              # [*, seq, 1]
        result = (r * H).sum(dim=-2)                             # [*, dim]
        
        # For completely masked rows, return zero (already all zero if no valid positions)
        if mask is not None:
            result = result.masked_fill(all_masked, 0.0)
        
        return result


class FuseGate(nn.Module):
    """Equation (5) in the paper – element-wise gated fusion of two vectors."""

    def __init__(self, dim: int):
        super().__init__()
        self.Wf1 = nn.Linear(2 * dim, dim, bias=False)
        self.Wf2 = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """x, y : [*, dim]  ->  [*, dim]"""
        cat = torch.cat([x, y], dim=-1)
        g   = torch.sigmoid(self.Wf1(cat))
        z   = torch.tanh(self.Wf2(cat))
        return g * z


class FuseGateAttention(nn.Module):
    """
    Enhanced fusion mechanism using multi-head attention.
    Better than simple sigmoid gating for learning complex signal combinations.
    
    Given multiple input signals, learns to weight them adaptively using attention.
    """
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query, Key, Value projections (for single vector as query)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.Wo = nn.Linear(dim, dim, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(config.DROPOUT)
    
    def forward(self, primary: torch.Tensor, *secondary: torch.Tensor):
        """
        Fuse primary signal with secondary signals using attention.
        
        Args:
            primary: [*, dim] - primary signal (queries)
            *secondary: variable number of [*, dim] tensors (keys/values)
        
        Returns:
            fused: [*, dim] - fused representation
        """
        if not secondary:
            return primary
        
        # Stack all signals to attend over
        signals = torch.stack([primary] + list(secondary), dim=-2)  # [*, 1+K, dim]
        K = signals.shape[-2]
        
        # Project to Q, K, V
        Q = self.Wq(primary).unsqueeze(-2)  # [*, 1, dim]
        K = self.Wk(signals).view(*signals.shape[:-1], self.num_heads, self.head_dim)  # [*, K, H, hd]
        V = self.Wv(signals).view(*signals.shape[:-1], self.num_heads, self.head_dim)  # [*, K, H, hd]
        
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim)  # [*, 1, H, hd]
        
        # Multi-head attention
        scores = (Q * K).sum(-1) / (self.head_dim ** 0.5)  # [*, K, H]
        attn = F.softmax(scores, dim=1)  # [*, K, H]
        attn = self.dropout(attn)
        
        # Weighted aggregation
        agg = (attn.unsqueeze(-1) * V).sum(dim=1)  # [*, H, hd]
        agg = agg.reshape(*agg.shape[:-2], self.dim)  # [*, dim]
        
        # Output projection and residual
        fused = self.norm(primary + self.Wo(agg))
        
        return fused


# ── Title Encoder  (Section 4.2) ───────────────────────────────────────────────

class TitleEncoder(nn.Module):
    """
    Multi-head self-attention + gated aggregation on word embeddings.
    Produces one d-dim vector per news title.
    """

    def __init__(self, word_emb_matrix: torch.Tensor, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        vocab_size, word_dim = word_emb_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            word_emb_matrix, freeze=False, padding_idx=0
        )
        # Project word_dim -> hidden_dim if they differ
        self.input_proj = nn.Linear(word_dim, hidden_dim, bias=False) if word_dim != hidden_dim else nn.Identity()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.agg     = GatedAggregation(hidden_dim)

    def forward(self, title_ids: torch.Tensor):
        """
        title_ids : [*, MAX_TITLE]   (can be any leading batch dims)
        Returns   : [*, hidden_dim]
        """
        leading = title_ids.shape[:-1]
        L = title_ids.shape[-1]
        flat_ids = title_ids.reshape(-1, L)                       # [N, L]

        x = self.dropout(self.input_proj(self.embedding(flat_ids)))  # [N, L, d]
        key_padding_mask = (flat_ids == 0)                          # [N, L]
        
        # Zero out padding positions to avoid numerical issues with attention masking
        x = x * (~key_padding_mask).float().unsqueeze(-1)           # [N, L, d]

        # Use attention without key_padding_mask for numerical stability
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=None)  # Don't use masked attention
        x = self.norm(x + self.dropout(attn_out))                   # residual + LN

        valid_mask = ~key_padding_mask                               # [N, L]
        h = self.agg(x, valid_mask)                                  # [N, d]
        
        return h.reshape(*leading, -1)                               # [*, d]


# ── Interest Encoder  (Section 4.3) ────────────────────────────────────────────

class GraphTransformerLayer(nn.Module):
    """
    Multi-head graph transformer (Equations 3-5 from original GREP paper).
    Used for both Potential Interest Encoding and Bi-directional Interaction.
    Captures multiple aspects of neighbor relationships through multi-head attention.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        # Multi-head projections for rich attention patterns
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        
        # Output layer norm for stability
        self.norm = nn.LayerNorm(dim)
        
        # Gated fusion for combining original and aggregated
        self.fuse = FuseGate(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_emb: torch.Tensor,         # [N, d]
        neighbor_ids: torch.Tensor,     # [N, K]   indices into node_emb
        neighbor_mask: torch.Tensor,    # [N, K]   BoolTensor True=valid
    ):
        """Returns updated node embeddings [N, d]."""
        N, d = node_emb.shape
        K    = neighbor_ids.shape[1]

        # Gather neighbor embeddings  [N, K, d]
        safe_ids = neighbor_ids.clamp(0, N - 1)
        nbr_emb  = node_emb[safe_ids]              # [N, K, d]

        # Multi-head attention  (Equation 3 in paper)
        Q = self.Wq(node_emb).view(N, 1,  self.num_heads, self.head_dim)  # [N,1,H,hd]
        K_ = self.Wk(nbr_emb).view(N, K, self.num_heads, self.head_dim)   # [N,K,H,hd]
        V_ = self.Wv(nbr_emb).view(N, K, self.num_heads, self.head_dim)   # [N,K,H,hd]

        # scores  [N, K, H]
        scores = (Q * K_).sum(-1) / (self.head_dim ** 0.5)
        # mask padding with large negative instead of -inf to avoid NaN gradients
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(-1), float("-1e4"))
        attn = F.softmax(scores, dim=1)                                    # [N, K, H]
        attn = self.dropout(attn)

        # aggregate  [N, H, hd] -> [N, d]
        agg = (attn.unsqueeze(-1) * V_).sum(dim=1)                         # [N, H, hd]
        agg = agg.reshape(N, d)                                            # [N, d]

        # Handle rows with no valid neighbors
        no_nbr = ~neighbor_mask.any(dim=1, keepdim=True)  # [N, 1]
        agg    = agg.masked_fill(no_nbr, 0.0)

        # Layer norm + gated fusion for better signal propagation
        return self.fuse(node_emb, self.norm(agg))                        # [N, d]


class InterestEncodingBlock(nn.Module):
    """
    One block of the Interest Encoder (Section 4.3):
      1. Existing Interest Encoding  – multi-head self-attn on clicked-title reps
      2. Potential Interest Encoding – Multi-head Graph Transformer on entity KG
      3. Bi-directional Interaction  – Graph Transformer on title-entity bipartite graph
    """

    def __init__(self, hidden_dim: int, kg_heads: int, bi_heads: int, dropout: float):
        super().__init__()
        self.norm_exist = nn.LayerNorm(hidden_dim)
        self.exist_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=kg_heads,
            dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        self.potential_gt = GraphTransformerLayer(hidden_dim, kg_heads, dropout)
        self.fuse_title   = FuseGate(hidden_dim)

    def forward(
        self,
        title_reps:    torch.Tensor,   # [B, n_hist, d]
        entity_reps:   torch.Tensor,   # [E, d]   all entity embeddings (shared)
        hist_ent_ids:  torch.Tensor,   # [B, n_hist, max_ent]
        hist_ent_mask: torch.Tensor,   # [B, n_hist, max_ent]
        hist_mask:     torch.Tensor,   # [B, n_hist]  True=valid
        kg_nbr_ids:    torch.Tensor,   # [E, K]
        kg_nbr_mask:   torch.Tensor,   # [E, K]
        cand_ent_ids:  torch.Tensor = None,   # [B, n_cand, max_ent]  optional for sparse computation
        cand_ent_mask: torch.Tensor = None,   # [B, n_cand, max_ent]
    ):
        B, n, d = title_reps.shape

        # ── 1. Existing Interest Encoding ──────────────────────────────────
        title_reps_safe = title_reps * hist_mask.float().unsqueeze(-1)
        attn_out, _ = self.exist_attn(
            title_reps_safe, title_reps_safe, title_reps_safe,
            key_padding_mask=None
        )
        title_reps = self.norm_exist(title_reps + self.dropout(attn_out))  # [B, n, d]

        # ── 2. Potential Interest Encoding with Sparse Entity Computation ──
        # Collect unique entity IDs from batch + their KG neighbors for sparse GT
        batch_ent_ids = torch.cat([
            hist_ent_ids.reshape(-1),
            cand_ent_ids.reshape(-1) if cand_ent_ids is not None else torch.tensor([], dtype=hist_ent_ids.dtype, device=hist_ent_ids.device),
        ]).unique()
        batch_ent_ids = batch_ent_ids[batch_ent_ids > 0]  # Remove padding (id 0)
        
        # Collect KG neighbors of batch entities
        kg_nbr_of_batch = kg_nbr_ids[batch_ent_ids]  # [|batch_ent|, K]
        kg_nbr_flat = kg_nbr_of_batch[kg_nbr_of_batch > 0].unique()  # Remove 0s
        
        # Union: batch entities + their KG neighbors
        sparse_ent_ids = torch.cat([batch_ent_ids, kg_nbr_flat]).unique()
        sparse_ent_ids = sparse_ent_ids[sparse_ent_ids > 0]
        
        # Create mapping: original_id -> sparse_idx
        E_total = entity_reps.shape[0]
        id_to_sparse = torch.full((E_total,), -1, dtype=torch.long, device=entity_reps.device)
        id_to_sparse[sparse_ent_ids] = torch.arange(len(sparse_ent_ids), dtype=torch.long, device=entity_reps.device)
        
        # Extract sparse entity representations
        sparse_entity_reps = entity_reps[sparse_ent_ids]  # [|sparse|, d]
        
        # Map KG neighbors to sparse indices
        sparse_kg_nbr_ids = id_to_sparse[kg_nbr_ids[sparse_ent_ids]].clamp(0, len(sparse_ent_ids) - 1)  # [|sparse|, K]
        sparse_kg_nbr_mask = kg_nbr_mask[sparse_ent_ids]  # [|sparse|, K]
        
        # Run GT on sparse subset
        sparse_entity_reps = self.potential_gt(sparse_entity_reps, sparse_kg_nbr_ids, sparse_kg_nbr_mask)  # [|sparse|, d]
        
        # Scatter results back to full entity table (preserve dtype)
        entity_reps = entity_reps.clone()  # [E, d]
        entity_reps[sparse_ent_ids] = sparse_entity_reps.to(entity_reps.dtype)  # Ensure dtype match

        # ── 3. Bi-directional Interaction ──────────────────────────────────
        # Fully batched implementation (no Python loop)
        E_size = entity_reps.shape[0]
        safe_e_ids = hist_ent_ids.clamp(0, E_size - 1)                    # [B, n, max_ent]
        
        # Gather entity embeddings for each (batch, title) position
        # entity_reps: [E, d] -> index with [B, n, max_ent] -> [B, n, max_ent, d]
        nbr_ent_emb = entity_reps[safe_e_ids]                             # [B, n, max_ent, d]
        
        # Title attention over entity neighbors
        t_q = title_reps.unsqueeze(2)                                     # [B, n, 1, d]
        scores = (t_q * nbr_ent_emb).sum(-1) / (d ** 0.5)                # [B, n, max_ent]
        scores = scores.masked_fill(~hist_ent_mask, float("-1e4"))       # [B, n, max_ent]
        attn   = F.softmax(scores, dim=-1).unsqueeze(-1)                 # [B, n, max_ent, 1]
        
        # Aggregate entity info per title
        agg_e  = (attn * nbr_ent_emb).sum(dim=2)                          # [B, n, d]
        
        # Mask rows with no entities
        no_ent = ~hist_ent_mask.any(dim=2, keepdim=True)                 # [B, n, 1]
        agg_e = agg_e.masked_fill(no_ent, 0.0)                           # [B, n, d]
        
        # Fuse for each sample in batch (vectorized via broadcasting in FuseGate)
        title_reps = self.fuse_title(title_reps, agg_e)                  # [B, n, d]

        return title_reps, entity_reps


# ── GREP  (Section 4) ──────────────────────────────────────────────────────────

class GREP(nn.Module):
    def __init__(
        self,
        word_emb_matrix:   torch.Tensor,    # [V, word_dim]
        entity_emb_matrix: torch.Tensor,    # [E, entity_dim]
        kg_nbr_ids:        torch.Tensor,    # [E, K]
        kg_nbr_mask:       torch.Tensor,    # [E, K]
    ):
        super().__init__()
        d        = config.HIDDEN_DIM
        dropout  = config.DROPOUT

        # Entity embedding table (trainable)
        E, ent_dim = entity_emb_matrix.shape
        self.entity_embedding = nn.Embedding.from_pretrained(
            entity_emb_matrix.float(), freeze=False, padding_idx=0
        )
        self.entity_proj = nn.Linear(ent_dim, d, bias=False) if ent_dim != d else nn.Identity()

        # Pre-computed KG neighbor tables (not parameters)
        self.register_buffer("kg_nbr_ids",  kg_nbr_ids)
        self.register_buffer("kg_nbr_mask", kg_nbr_mask)

        # Title encoder
        self.title_encoder = TitleEncoder(
            word_emb_matrix=word_emb_matrix,
            hidden_dim=d,
            num_heads=config.TITLE_HEAD_NUM,
            dropout=dropout,
        )

        # Interest encoder – L stacked blocks
        self.interest_blocks = nn.ModuleList([
            InterestEncodingBlock(d, config.KG_HEAD_NUM, config.BI_HEAD_NUM, dropout)
            for _ in range(config.INTEREST_LAYER_NUM)
        ])

        # Final aggregation
        self.final_exist_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.KG_HEAD_NUM,
            dropout=dropout, batch_first=True
        )
        self.final_exist_norm = nn.LayerNorm(d)
        self.exist_agg   = GatedAggregation(d)
        self.potential_agg = GatedAggregation(d)

        # Prediction gate (Eq. 8 & 9)
        self.user_gate = nn.Linear(2 * d, 1, bias=True)
        self.news_gate = nn.Linear(2 * d, 1, bias=True)

        self.dropout = nn.Dropout(dropout)

    # ── encode a batch of news into (title_rep, entity_rep) ───────────────
    def _encode_titles(self, title_ids: torch.Tensor) -> torch.Tensor:
        """title_ids: [B, N, L]  ->  [B, N, d]"""
        return self.title_encoder(title_ids)

    def _get_entity_reps(self) -> torch.Tensor:
        """Return projected entity embedding matrix [E, d]."""
        return self.entity_proj(self.dropout(self.entity_embedding.weight))
    
    def _compute_temporal_decay_weights(self, hist_lens: list, device: torch.device) -> torch.Tensor:
        """
        Compute temporal decay weights for history items.
        More recent items (later in history) get higher weights.
        
        Args:
            hist_lens: list of actual history lengths per sample in batch
            device: torch device
        
        Returns:
            Tensor [B, MAX_HIST] with temporal decay weights
        """
        B = len(hist_lens)
        max_hist = config.MAX_HISTORY_LEN
        
        # Create position-based decay: older positions (left) get lower weight
        # Position 0 (oldest) → weight ≈ exp(-λ * max_hist)
        # Position max_hist-1 (newest) → weight ≈ exp(0) = 1.0
        positions = torch.arange(max_hist, dtype=torch.float32, device=device).unsqueeze(0)  # [1, MAX_HIST]
        
        # Decay: weight[i] = exp(-λ * (max_hist - i))
        # So recent items (high i) have weight close to 1, old items have weight < 1
        decay_weights = torch.exp(-config.TEMPORAL_DECAY_LAMBDA * (max_hist - positions))  # [1, MAX_HIST]
        
        # Apply actual history length mask
        weights = torch.ones(B, max_hist, dtype=torch.float32, device=device)
        for b, h_len in enumerate(hist_lens):
            weights[b, :h_len] = decay_weights[0, :h_len]
            weights[b, h_len:] = 0.0  # Zero out padding
        
        return weights

    # ── user representation ────────────────────────────────────────────────
    def _encode_user(
        self,
        hist_titles:   torch.Tensor,   # [B, n, L]
        hist_ent_ids:  torch.Tensor,   # [B, n, max_ent]
        hist_ent_mask: torch.Tensor,   # [B, n, max_ent]
        hist_lens:     list,
        cand_ent_ids:  torch.Tensor = None,   # [B, c, max_ent]  for sparse computation
        cand_ent_mask: torch.Tensor = None,   # [B, c, max_ent]
    ):
        B, n, _ = hist_titles.shape
        d = config.HIDDEN_DIM

        # history validity mask  [B, n]
        hist_mask = torch.zeros(B, n, dtype=torch.bool, device=hist_titles.device)
        for b, hl in enumerate(hist_lens):
            hist_mask[b, :hl] = True

        # ── Temporal decay weights (position-based for recent news) ────────
        temporal_weights = self._compute_temporal_decay_weights(hist_lens, hist_titles.device)  # [B, n]
        # Combine temporal weight with validity mask
        temporal_mask = hist_mask.float() * temporal_weights  # [B, n]

        # Initial title representations  [B, n, d]
        title_reps = self._encode_titles(hist_titles)

        # Initial entity representations  [E, d]
        entity_reps = self._get_entity_reps()

        # Stack L interest encoding blocks
        for block in self.interest_blocks:
            title_reps, entity_reps = block(
                title_reps, entity_reps,
                hist_ent_ids, hist_ent_mask,
                hist_mask,
                self.kg_nbr_ids, self.kg_nbr_mask,
                cand_ent_ids, cand_ent_mask,
            )

        # ── Existing interest representation (Section 4.4.1) ───────────────
        key_pad = ~hist_mask
        # Zero out non-valid positions to avoid numerical issues
        title_reps_safe = title_reps * hist_mask.float().unsqueeze(-1)
        attn_out, _ = self.final_exist_attn(
            title_reps_safe, title_reps_safe, title_reps_safe,
            key_padding_mask=None
        )
        title_reps = self.final_exist_norm(title_reps + self.dropout(attn_out))
        # Apply temporal weighting to prioritize recent news before aggregation
        title_reps_weighted = title_reps * temporal_mask.unsqueeze(-1)  # [B, n, d]
        u_h = self.exist_agg(title_reps_weighted, hist_mask)   # [B, d]

        # ── Potential interest representation (Section 4.4.2) ──────────────
        # Collect all entity indices appearing in user history  [B, n*max_ent]
        B2, n2, me = hist_ent_ids.shape
        flat_ids   = hist_ent_ids.reshape(B2, n2 * me)        # [B, n*me]
        flat_mask  = hist_ent_mask.reshape(B2, n2 * me)       # [B, n*me]

        # Gather current entity reps for each user's entities  [B, n*me, d]
        E_size    = entity_reps.shape[0]
        safe_ids  = flat_ids.clamp(0, E_size - 1)
        ent_seq   = entity_reps[safe_ids]                     # [B, n*me, d]

        u_e = self.potential_agg(ent_seq, flat_mask)          # [B, d]

        # ── Combine (Eq. 8) ────────────────────────────────────────────────
        alpha = torch.sigmoid(self.user_gate(torch.cat([u_h, u_e], dim=-1)))  # [B,1]
        u = alpha * u_h + (1 - alpha) * u_e                                   # [B, d]

        return u, entity_reps

    # ── candidate news representation ─────────────────────────────────────
    def _encode_candidate(
        self,
        cand_titles:  torch.Tensor,   # [B, C, L]
        cand_ent_ids: torch.Tensor,   # [B, C, max_ent]
        cand_ent_mask: torch.Tensor,  # [B, C, max_ent]
        entity_reps:  torch.Tensor,   # [E, d]
    ):
        B, C, L = cand_titles.shape
        d = config.HIDDEN_DIM

        c_h = self._encode_titles(cand_titles)                # [B, C, d]

        # Entity agg per candidate
        E_size   = entity_reps.shape[0]
        safe_ids = cand_ent_ids.clamp(0, E_size - 1)         # [B, C, me]
        ent_seq  = entity_reps[safe_ids]                      # [B, C, me, d]

        scores = (c_h.unsqueeze(-2) * ent_seq).sum(-1) / (d ** 0.5)  # [B, C, me]
        scores = scores.masked_fill(~cand_ent_mask, float("-1e4"))
        attn   = F.softmax(scores, dim=-1).unsqueeze(-1)               # [B, C, me, 1]
        no_ent = ~cand_ent_mask.any(dim=-1, keepdim=True)              # [B, C, 1]
        c_e    = (attn * ent_seq).sum(dim=-2)                          # [B, C, d]
        c_e    = c_e.masked_fill(no_ent, 0.0)

        # Combine (Eq. 9)
        cat    = torch.cat([c_h, c_e], dim=-1)                        # [B, C, 2d]
        alpha  = torch.sigmoid(self.news_gate(cat))                   # [B, C, 1]
        c      = alpha * c_h + (1 - alpha) * c_e                      # [B, C, d]
        return c

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, batch: dict):
        u, entity_reps = self._encode_user(
            batch["hist_titles"],
            batch["hist_ent_ids"],
            batch["hist_ent_mask"],
            batch["hist_lens"],
            batch["cand_ent_ids"],
            batch["cand_ent_mask"],
        )

        c = self._encode_candidate(
            batch["cand_titles"],
            batch["cand_ent_ids"],
            batch["cand_ent_mask"],
            entity_reps,
        )

        # Dot-product click probability  [B, C]
        scores = (u.unsqueeze(1) * c).sum(-1)
        return scores
