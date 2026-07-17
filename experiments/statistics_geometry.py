import os
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    permutation_test,
)

from config import *

########################################################################
# Bootstrap
########################################################################

def bootstrap_ci(
    values,
    n_boot=10000,
    ci=95,
):

    values = np.asarray(values)

    rng = np.random.default_rng(SEED)

    means = np.zeros(n_boot)

    for i in range(n_boot):

        sample = rng.choice(

            values,

            size=len(values),

            replace=True,

        )

        means[i] = sample.mean()

    alpha = (100 - ci) / 2

    lower = np.percentile(
        means,
        alpha,
    )

    upper = np.percentile(
        means,
        100 - alpha,
    )

    return (

        means.mean(),

        lower,

        upper,

    )


########################################################################
# Cohen's d
########################################################################

def cohens_d(
    x,
    y,
):

    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    dof = nx + ny - 2

    pooled = np.sqrt(

        (

            (nx - 1) * x.var(ddof=1)

            +

            (ny - 1) * y.var(ddof=1)

        )

        / dof

    )

    return (

        x.mean()

        -

        y.mean()

    ) / (

        pooled

        +

        1e-12

    )


########################################################################
# Cliff Delta
########################################################################

def cliffs_delta(
    x,
    y,
):

    x = np.asarray(x)
    y = np.asarray(y)

    greater = 0
    lower = 0

    for a in x:

        greater += np.sum(a > y)

        lower += np.sum(a < y)

    return (

        greater - lower

    ) / (

        len(x)

        * len(y)

    )


########################################################################
# Hedges g
########################################################################

def hedges_g(
    x,
    y,
):

    d = cohens_d(
        x,
        y,
    )

    correction = (

        1

        -

        3

        /

        (

            4 * (len(x) + len(y))

            - 9

        )

    )

    return d * correction


########################################################################
# Permutation Test
########################################################################

def permutation_pvalue(
    x,
    y,
):

    result = permutation_test(

        (

            x,

            y,

        ),

        statistic=lambda a, b: np.mean(a) - np.mean(b),

        permutation_type="independent",

        n_resamples=5000,

        alternative="two-sided",

        random_state=SEED,

    )

    return result.pvalue


########################################################################
# Layer Statistics
########################################################################

def analyse_layers():

    layer = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_layers.csv",

        )

    )

    rows = []

    for _, r in layer.iterrows():

        rows.append(

            {

                "layer":

                    int(r.layer),

                "cosine":

                    r.cosine_mean,

                "cka":

                    r.cka_mean,

                "shuffle":

                    r.shuffle_mean,

            }

        )

    return pd.DataFrame(rows)


########################################################################
# Global Statistics
########################################################################

def analyse_geometry():

    pair = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_pairwise.csv",

        )

    )

    cosine = pair.cosine_last.values

    cka = pair.cka_last.values

    shuffle = pair.shuffle_last.values

    ####################################################################
    # Confidence Interval
    ####################################################################

    cosine_ci = bootstrap_ci(
        cosine
    )

    cka_ci = bootstrap_ci(
        cka
    )

    shuffle_ci = bootstrap_ci(
        shuffle
    )

    ####################################################################
    # Statistical Tests
    ####################################################################

    t_cosine = ttest_rel(
        cosine,
        shuffle,
    )

    w_cosine = wilcoxon(
        cosine,
        shuffle,
    )

    p_perm = permutation_pvalue(
        cosine,
        shuffle,
    )

    ####################################################################
    # Effect Size
    ####################################################################

    d = cohens_d(
        cosine,
        shuffle,
    )

    g = hedges_g(
        cosine,
        shuffle,
    )

    delta = cliffs_delta(
        cosine,
        shuffle,
    )

    summary = pd.DataFrame(

        [

            {

                "metric":

                    "Cosine",

                "mean":

                    cosine.mean(),

                "std":

                    cosine.std(),

                "ci_lower":

                    cosine_ci[1],

                "ci_upper":

                    cosine_ci[2],

            },

            {

                "metric":

                    "CKA",

                "mean":

                    cka.mean(),

                "std":

                    cka.std(),

                "ci_lower":

                    cka_ci[1],

                "ci_upper":

                    cka_ci[2],

            },

            {

                "metric":

                    "Shuffle",

                "mean":

                    shuffle.mean(),

                "std":

                    shuffle.std(),

                "ci_lower":

                    shuffle_ci[1],

                "ci_upper":

                    shuffle_ci[2],

            },

        ]

    )

    summary.to_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_statistics.csv",

        ),

        index=False,

    )

    hypothesis = pd.DataFrame(

        [

            {

                "paired_t":

                    t_cosine.statistic,

                "paired_t_p":

                    t_cosine.pvalue,

                "wilcoxon":

                    w_cosine.statistic,

                "wilcoxon_p":

                    w_cosine.pvalue,

                "permutation_p":

                    p_perm,

                "cohens_d":

                    d,

                "hedges_g":

                    g,

                "cliffs_delta":

                    delta,

            }

        ]

    )

    hypothesis.to_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_hypothesis.csv",

        ),

        index=False,

    )

    return (

        summary,

        hypothesis,

    )


########################################################################
# Entry
########################################################################

def run_geometry_statistics():

    analyse_layers()

    return analyse_geometry()


if __name__ == "__main__":

    run_geometry_statistics()
