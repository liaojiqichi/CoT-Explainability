import random
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

######################################################
# Seed
######################################################

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

######################################################
# Shuffle CoT
######################################################

def shuffle_cot(cot):

    steps = [
        x
        for x in cot.split("\n")
        if len(x.strip()) > 0
    ]

    random.shuffle(steps)

    return "\n".join(steps)

######################################################
# PCA
######################################################

def project_pca(hidden):

    pca = PCA(n_components=2)

    return pca.fit_transform(hidden)

######################################################
# TSNE
######################################################

def project_tsne(hidden):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    return tsne.fit_transform(hidden)

######################################################
# Normalize
######################################################

def normalize(x):

    x = np.maximum(
        np.array(x),
        0
    )

    return x / (
        np.sum(x) + 1e-12
    )
