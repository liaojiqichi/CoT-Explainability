"""
Global configuration
ACL Mechanistic Equifinality
"""

import os
import random
import numpy as np
import torch

###########################################################
# Model
###########################################################

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

###########################################################
# Dataset
###########################################################

DATASET = "gsm8k"

SAMPLE_SIZE = 200

SEED = 42

###########################################################
# Generation
###########################################################

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.0

TOP_P = 1.0

###########################################################
# Prompt
###########################################################

PREFIX_ALGEBRA = (
    "Solve step by step using algebra. "
    "Let x be unknown. "
)

PREFIX_ARITHMETIC = (
    "Solve step by step using arithmetic only. "
)

FINAL_PHRASE = "\nFinal answer: "

###########################################################
# Analysis
###########################################################

SINK_TOKENS = 5

BOOTSTRAP_ITER = 10000

PERMUTATION_ITER = 5000

NUM_PCA_COMPONENTS = 2

###########################################################
# Paths
###########################################################

ROOT = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(ROOT, "cache")

RESULT_DIR = os.path.join(ROOT, "results")

FIGURE_DIR = os.path.join(ROOT, "figures")

LOG_DIR = os.path.join(ROOT, "logs")

for path in [
    CACHE_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    LOG_DIR,
]:
    os.makedirs(path, exist_ok=True)

CACHE_FILE = os.path.join(
    CACHE_DIR,
    "cached_cots.csv"
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "experiment_results.csv"
)

###########################################################
# Random Seed
###########################################################

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
