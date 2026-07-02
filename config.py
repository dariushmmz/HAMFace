# config.py — central configuration for all paths and hyperparameters

import os

# ── Raw dataset (one sub-folder per identity, images inside) ──────────────────
RAW_FACES_DIR = ".dataset/raw_faces"

# ── Processed dataset paths ────────────────────────────────────────────────────
DATASET_ROOT = ".dataset/processed"
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
TEST_DIR     = os.path.join(DATASET_ROOT, "test")

# Fraction of each person's images reserved for testing
TEST_SPLIT = 0.2

TRAIN_DATA_PATH   = os.path.join(DATASET_ROOT, "train_data.npy")
TRAIN_LABELS_PATH = os.path.join(DATASET_ROOT, "train_labels.npy")
TEST_DATA_PATH    = os.path.join(DATASET_ROOT, "test_data.npy")
TEST_LABELS_PATH  = os.path.join(DATASET_ROOT, "test_labels.npy")

# ── Gallery paths ──────────────────────────────────────────────────────────────
GALLERY_DIR      = DATASET_ROOT
GALLERY_FULL_PKL = os.path.join(GALLERY_DIR, "gallery_full.pkl")
GALLERY_AVG_PKL  = os.path.join(GALLERY_DIR, "gallery_avg.pkl")

# ── Checkpoint paths ───────────────────────────────────────────────────────────
CHECKPOINT_DIR          = "checkpoints"
MODEL_WEIGHTS_PATH      = os.path.join(CHECKPOINT_DIR, "best_model.pt")
CLASS_WEIGHTS_PATH      = os.path.join(CHECKPOINT_DIR, "hamface_class_weights.npy")
YOLO_WEIGHTS_PATH       = os.path.join(CHECKPOINT_DIR, "yolov12n-face.pt")
YOLO_CONF_THRESHOLD     = 0.3

# ── Model / training hyperparameters ──────────────────────────────────────────
IMAGE_SIZE   = 128          # spatial resolution fed to the model
N_CLASSES    = 4           # number of identity classes
EMBED_DIM    = 128          # final embedding dimension

# HAMFace loss hyperparameters
LOSS_SCALE   = 30.0         # s  — logit scale
LOSS_MARGIN  = 0.5          # m  — base angular margin
LOSS_HARDNESS= 0.3          # t  — hardness-aware coefficient

# CvT stage hyperparameters
CVT_EMBED_DIM = 64
CVT_PATCH_SIZE = 4       # downsample factor before attention (reduces seq len 64x64→16x16)
CVT_NUM_HEADS = 1
CVT_FF_DIM    = 128
CVT_DROPOUT   = 0.1

# Channel-attention reduction ratio
CA_REDUCTION_RATIO = 16

# Training
BATCH_SIZE    = 8           # reduced to avoid OOM on CPU (was 32)
EPOCHS        = 5
LEARNING_RATE = 1e-3
RANDOM_SEED   = 42

# Augmentation (probability that blur / noise / contrast are applied)
AUGMENT_PROB  = 0.5         # roughly maps to random.choice([True, False, False])

# Gaussian blur
BLUR_KERNEL   = 5

# Gaussian noise
NOISE_MEAN    = 0.0
NOISE_STD     = 0.05

# Contrast adjustment
CONTRAST_ALPHA = 1.3
CONTRAST_BETA  = 0.0