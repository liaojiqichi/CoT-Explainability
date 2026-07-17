import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

########################################################################
# Global Plot Style
########################################################################

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.fontsize"] = 11

GEOMETRY_DIR = os.path.join(FIGURE_DIR, "geometry")
os.makedirs(GEOMETRY_DIR, exist_ok=True)


########################################################################
# Figure 4
# Layer-wise Cosine
########################################################################

def plot_layer_cosine():

    df = pd.read_csv(

        os.path.join(
            RESULT_DIR,
            "geometry_layers.csv"
        )

    )

    plt.figure(figsize=(8,5))

    plt.plot(

        df["layer"],
        df["cosine_mean"],
        linewidth=2,
        label="Cosine"

    )

    plt.fill_between(

        df["layer"],

        df["cosine_mean"]-df["cosine_std"],

        df["cosine_mean"]+df["cosine_std"],

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Cosine Similarity")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig4_layer_cosine.pdf"

        )

    )

    plt.close()


########################################################################
# Figure 5
# Layer-wise CKA
########################################################################

def plot_layer_cka():

    df = pd.read_csv(

        os.path.join(
            RESULT_DIR,
            "geometry_layers.csv"
        )

    )

    plt.figure(figsize=(8,5))

    plt.plot(

        df["layer"],

        df["cka_mean"],

        linewidth=2,

        label="CKA"

    )

    plt.fill_between(

        df["layer"],

        df["cka_mean"]-df["cka_std"],

        df["cka_mean"]+df["cka_std"],

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Linear CKA")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig5_layer_cka.pdf"

        )

    )

    plt.close()


########################################################################
# Figure 6
# Shuffle Baseline
########################################################################

def plot_shuffle():

    df = pd.read_csv(

        os.path.join(
            RESULT_DIR,
            "geometry_layers.csv"
        )

    )

    plt.figure(figsize=(8,5))

    plt.plot(

        df["layer"],

        df["cosine_mean"],

        linewidth=2,

        label="Real"

    )

    plt.plot(

        df["layer"],

        df["shuffle_mean"],

        "--",

        linewidth=2,

        label="Shuffle"

    )

    plt.xlabel("Layer")

    plt.ylabel("Cosine")

    plt.legend()

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig6_shuffle.pdf"

        )

    )

    plt.close()


########################################################################
# Figure 7
# Cross Query Baseline
########################################################################

def plot_cross_query():

    df = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "cross_query.csv"

        )

    )

    plt.figure(figsize=(8,5))

    for prompt in sorted(df.prompt.unique()):

        tmp = df[

            df.prompt == prompt

        ]

        plt.plot(

            tmp.layer,

            tmp.cosine,

            linewidth=1,

            alpha=.6,

            label=prompt

        )

    plt.xlabel("Layer")

    plt.ylabel("Cross-query Cosine")

    plt.grid(alpha=.3)

    plt.legend(

        fontsize=8,

        ncol=2

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig7_cross_query.pdf"

        )

    )

    plt.close()


########################################################################
# Figure 8
# Representation Collapse
########################################################################

def plot_collapse():

    df = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_pairwise.csv"

        )

    )

    collapse = np.concatenate(

        [

            df.collapse_a.values,

            df.collapse_b.values

        ]

    )

    plt.figure(figsize=(6,5))

    plt.hist(

        collapse,

        bins=30

    )

    plt.xlabel(

        "Collapse Score"

    )

    plt.ylabel(

        "Frequency"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig8_collapse.pdf"

        )

    )

    plt.close()


########################################################################
# Figure 9
# Layer-wise Drift
########################################################################

def plot_representation_drift():

    df = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_layers.csv"

        )

    )

    drift = df["layer"].values[:-1]

    values = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "geometry_layers.csv"

        )

    )

    plt.figure(figsize=(8,5))

    plt.plot(

        drift,

        values["cosine_mean"].values[:-1]

    )

    plt.xlabel("Layer")

    plt.ylabel("Representation Stability")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GEOMETRY_DIR,

            "fig9_representation_drift.pdf"

        )

    )

    plt.close()


########################################################################
# Run All
########################################################################

def run_geometry_plot():

    plot_layer_cosine()

    plot_layer_cka()

    plot_shuffle()

    plot_cross_query()

    plot_collapse()

    plot_representation_drift()


if __name__ == "__main__":

    run_geometry_plot()
