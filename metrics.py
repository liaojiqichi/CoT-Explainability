import numpy as np

#########################################################
# Cosine
#########################################################

def cosine(a, b):

    a = np.asarray(a, dtype=np.float64)

    b = np.asarray(b, dtype=np.float64)

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-12
    )

#########################################################
# Linear CKA
#########################################################

def linear_cka(x, y):

    x = x.astype(np.float64)

    y = y.astype(np.float64)

    x = x - x.mean()

    y = y - y.mean()

    return (
        np.dot(x, y) ** 2
    ) / (
        np.dot(x, x)
        * np.dot(y, y)
        + 1e-12
    )

#########################################################
# TVD
#########################################################

def tvd(p, q):

    return np.sum(
        np.abs(p - q)
    ) / 2

#########################################################
# Entropy
#########################################################

def entropy(prob):

    prob = prob + 1e-12

    return -np.sum(
        prob * np.log(prob)
    )

#########################################################
# JSD
#########################################################

def jsd(p, q):

    p = np.maximum(p, 0)

    q = np.maximum(q, 0)

    p = p / (p.sum() + 1e-12)

    q = q / (q.sum() + 1e-12)

    m = (p + q) / 2

    return 0.5 * (
        np.sum(p * np.log(p / m + 1e-12))
        + np.sum(q * np.log(q / m + 1e-12))
    )

#########################################################
# Drift
#########################################################

def representation_drift(hidden_states):

    drift = []

    for i in range(len(hidden_states)-1):

        drift.append(
            np.linalg.norm(
                hidden_states[i+1]
                - hidden_states[i]
            )
        )

    return np.asarray(drift)
