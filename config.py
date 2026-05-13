import os

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    # TRAIN_DIR = "./data/MINDlarge_train/MINDlarge_train"
    # VAL_DIR   = "./data/MINDlarge_dev/MINDlarge_dev"

    TRAIN_DIR = "../data/MINDsmall_train"
    VAL_DIR   = "../data/MINDsmall_eval"

    TRAIN_NEWS_FILE      = os.path.join(TRAIN_DIR, "news.tsv")
    TRAIN_BEHAVIORS_FILE = os.path.join(TRAIN_DIR, "behaviors.tsv")
    VAL_NEWS_FILE        = os.path.join(VAL_DIR,   "news.tsv")
    VAL_BEHAVIORS_FILE   = os.path.join(VAL_DIR,   "behaviors.tsv")

    # Test set paths (MIND LARGE)
    TEST_DIR = "../data/MINDlarge_test/MINDlarge_test"
    TEST_NEWS_FILE        = os.path.join(TEST_DIR,  "news.tsv")
    TEST_BEHAVIORS_FILE   = os.path.join(TEST_DIR,  "behaviors.tsv")

    # Entity / relation embeddings (shared across splits in MIND-small)
    ENTITY_EMBEDDING_FILE   = os.path.join(TRAIN_DIR, "entity_embedding.vec")
    RELATION_EMBEDDING_FILE = os.path.join(TRAIN_DIR, "relation_embedding.vec")

    # ── Vocabulary / Embedding ─────────────────────────────────────────────
    WORD_EMBEDDING_DIM   = 300  # GloVe 840B 300d
    ENTITY_EMBEDDING_DIM = 100  # from .vec files
    HIDDEN_DIM           = 100  # d
    
    # ── GloVe Embeddings ───────────────────────────────────────────────────
    GLOVE_FILE = "../data/glove.840B.300d.txt" 

    MAX_VOCAB_SIZE = 50_000     # cap word vocab

    # ── Title Encoder ──────────────────────────────────────────────────────
    MAX_TITLE_LEN  = 20   # max words per title
    TITLE_HEAD_NUM = 5    # multi-head self-attention heads in title encoder
                          # (100 / 5 = 20 per head – must divide HIDDEN_DIM)

    # ── Interest Encoder ───────────────────────────────────────────────────
    INTEREST_LAYER_NUM = 2  # L  – number of stacked interest encoding blocks
    KG_HEAD_NUM        = 4  # Tp – heads in Graph Transformer (increased for entity modeling)
    BI_HEAD_NUM        = 2  # Tb – heads in bi-directional interaction

    # ── User history ───────────────────────────────────────────────────────
    MAX_HISTORY_LEN = 50    # max clicked news per user

    # ── Training ───────────────────────────────────────────────────────────
    NEG_SAMPLE_RATIO = 6    # k negative samples per positive (reduced for quality)
    HARD_NEG_RATIO   = 0.6  # fraction of hard negatives (same-category), rest random
    BATCH_SIZE       = 256
    VAL_BATCH_SIZE   = 64   # batch size for validation/testing
    LEARNING_RATE    = 1e-4
    NUM_EPOCHS       = 10   # Increased from 5: gentler curriculum (1.8% per epoch vs 3.5%)
    DROPOUT          = 0.3
    WEIGHT_DECAY     = 1e-4 # L2 regularization for preventing overfitting
    USE_BPR_LOSS     = False # use cross-entropy loss (more stable gradient spread)
    
    # ── Gradient Accumulation ───────────────────────────────────────────────
    ACCUMULATION_STEPS = 2  # accumulate gradients every N steps (effective batch = 128 * 2 = 256)
    
    # ── Learning Rate Scheduler ────────────────────────────────────────────
    WARMUP_STEPS     = 500      # linear warmup for first 500 steps
    LR_DECAY_FACTOR  = 0.95     # decay LR by this factor each epoch
    
    # ── Temporal Decay (Recency Weighting) ─────────────────────────────────
    # Weight recent news higher: weight = exp(-λ*age). Reduced from 0.01 to soften weighting
    TEMPORAL_DECAY_LAMBDA = 0.005

    # ── KG graph construction ──────────────────────────────────────────────
    # Built from entity co-occurrence in training news (as stated in paper)
    MAX_ENTITY_NEIGHBORS = 20   # max neighbors per entity node in KG

    # ── Data Augmentation ──────────────────────────────────────────────────
    USE_AUGMENTATION = True     # Enable data augmentation during training

    # ── Device ────────────────────────────────────────────────────────────
    DEVICE = "cuda"

    # ── Misc ──────────────────────────────────────────────────────────────
    SEED            = 42
    MODEL_SAVE_PATH = "checkpoints/latest_6/grep_best.pt"
    LOG_EVERY       = 300   # log training loss every N steps


config = Config()