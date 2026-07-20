import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import *
from extraction import load_signal

########################################################################
# Experiment
#
# Fig16
# PCA Trajectory
#
# Fig17
# UMAP / TSNE Trajectory
#
# Fig18
# Representation Drift Matrix
#
# Fig19
# Layer Curvature
#
# Fig20
# Representation Speed
########################################################################


class RepresentationTrajectoryExperiment:

    def __init__(self):

        self.index = pd.read_csv(

            os.path.join(

                CACHE_DIR,

                "cache_index.csv"

            )

        )

        self.prompts = list(

            PROMPT_FAMILIES.keys()

        )

        self.records = []

    ####################################################################
    # PCA
    ####################################################################

    def pca(

        self,

        layer_matrix,

    ):

        pca = PCA(

            n_components=2,

            random_state=SEED

        )

        return pca.fit_transform(

            layer_matrix

        )

    ####################################################################
    # TSNE
    ####################################################################

    def tsne(

        self,

        layer_matrix,

    ):

        perplexity = min(

            10,

            layer_matrix.shape[0]-1

        )

        tsne = TSNE(

            n_components=2,

            perplexity=perplexity,

            init="pca",

            random_state=SEED,

            learning_rate="auto"

        )

        return tsne.fit_transform(

            layer_matrix

        )

    ####################################################################
    # Drift Matrix
    ####################################################################

    def drift_matrix(

        self,

        hidden,

    ):

        n = hidden.shape[0]

        matrix = np.zeros(

            (

                n,

                n

            )

        )

        for i in range(n):

            for j in range(n):

                matrix[i,j] = np.linalg.norm(

                    hidden[i]

                    -

                    hidden[j]

                )

        return matrix

    ####################################################################
    # Speed
    ####################################################################

    def speed(

        self,

        hidden,

    ):

        return np.linalg.norm(

            np.diff(

                hidden,

                axis=0

            ),

            axis=1

        )

    ####################################################################
    # Curvature
    ####################################################################

    def curvature(

        self,

        hidden,

    ):

        velocity = np.diff(

            hidden,

            axis=0

        )

        acceleration = np.diff(

            velocity,

            axis=0

        )

        return np.linalg.norm(

            acceleration,

            axis=1

        )

    ####################################################################
    # Analyze
    ####################################################################

    def analyse(

        self,

        cache_file,

        sample,

        prompt,

    ):

        signal = load_signal(

            cache_file

        )

        hidden = signal[

            "layer_matrix"

        ]

        pca = self.pca(

            hidden

        )

        tsne = self.tsne(

            hidden

        )

        drift = self.drift_matrix(

            hidden

        )

        speed = self.speed(

            hidden

        )

        curvature = self.curvature(

            hidden

        )

        np.save(

            os.path.join(

                RESULT_DIR,

                f"trajectory_pca_{sample}_{prompt}.npy"

            ),

            pca

        )

        np.save(

            os.path.join(

                RESULT_DIR,

                f"trajectory_tsne_{sample}_{prompt}.npy"

            ),

            tsne

        )

        np.save(

            os.path.join(

                RESULT_DIR,

                f"drift_matrix_{sample}_{prompt}.npy"

            ),

            drift

        )

        self.records.append(

            {

                "sample":

                    sample,

                "prompt":

                    prompt,

                "speed":

                    speed,

                "curvature":

                    curvature,

                "path_length":

                    speed.sum(),

                "max_speed":

                    speed.max(),

                "mean_speed":

                    speed.mean(),

                "mean_curvature":

                    curvature.mean()

                    if len(curvature)

                    else 0,

            }

        )

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        for _, row in tqdm(

            self.index.iterrows(),

            total=len(

                self.index

            )

        ):

            sid = row["id"]

            for prompt in self.prompts:

                self.analyse(

                    row[prompt],

                    sid,

                    prompt

                )

        return self.save()

    ####################################################################
    # Save
    ####################################################################

    def save(self):

        df = pd.DataFrame(

            self.records

        )

        df.to_csv(

            os.path.join(

                RESULT_DIR,

                "representation_trajectory.csv"

            ),

            index=False

        )

        summary = (

            df

            .groupby(

                "prompt"

            )

            .agg(

                {

                    "path_length":["mean","std"],

                    "mean_speed":["mean","std"],

                    "max_speed":["mean","std"],

                    "mean_curvature":["mean","std"]

                }

            )

        )

        summary.to_csv(

            os.path.join(

                RESULT_DIR,

                "trajectory_summary.csv"

            )

        )

        return df


########################################################################
# Entry
########################################################################

def run_representation_trajectory():

    exp = RepresentationTrajectoryExperiment()

    return exp.run()


if __name__ == "__main__":

    run_representation_trajectory()
