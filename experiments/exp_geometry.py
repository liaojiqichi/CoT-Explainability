import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

from config import *
from metrics import *
from extraction import load_signal

########################################################################
# Geometry Experiment
#
# Figure 4
# Layer-wise Cosine Similarity
#
# Figure 5
# Layer-wise Linear CKA
#
# Figure 6
# Shuffle Baseline
#
# Figure 7
# Cross-query Baseline
#
# Figure 8
# Representation Collapse
########################################################################


class GeometryExperiment:

    def __init__(self):

        self.index = pd.read_csv(
            os.path.join(
                CACHE_DIR,
                "cache_index.csv"
            )
        )

        self.prompt_names = list(PROMPT_FAMILIES.keys())

        self.results = []

    ####################################################################
    # Utilities
    ####################################################################

    @staticmethod
    def normalize(v):

        v = np.asarray(v)

        return v / (
            np.linalg.norm(v) + 1e-12
        )

    ####################################################################
    # Load Layer Matrix
    ####################################################################

    def load_layer_matrix(
        self,
        cache_path,
    ):

        signal = load_signal(cache_path)

        return signal["layer_matrix"]

    ####################################################################
    # Cosine Layer Curve
    ####################################################################

    def cosine_curve(
        self,
        hidden_a,
        hidden_b,
    ):

        values = []

        for layer in range(hidden_a.shape[0]):

            values.append(

                cosine(

                    hidden_a[layer],

                    hidden_b[layer]

                )

            )

        return 1 - np.asarray(values)

    ####################################################################
    # Linear CKA Curve
    ####################################################################

    def cka_curve(
        self,
        hidden_a,
        hidden_b,
    ):

        values = []

        for layer in range(hidden_a.shape[0]):

            values.append(

                linear_cka(

                    hidden_a[layer],

                    hidden_b[layer]

                )

            )

        return np.asarray(values)

    ####################################################################
    # Representation Drift
    ####################################################################

    def drift_curve(
        self,
        hidden,
    ):

        drift = []

        for layer in range(

            hidden.shape[0] - 1

        ):

            drift.append(

                np.linalg.norm(

                    hidden[layer + 1]

                    -

                    hidden[layer]

                )

            )

        return np.asarray(drift)

    ####################################################################
    # Collapse Score
    ####################################################################

    def collapse_score(
        self,
        hidden,
    ):

        dist = []

        for i in range(

            hidden.shape[0]

        ):

            for j in range(

                i + 1,

                hidden.shape[0]

            ):

                dist.append(

                    np.linalg.norm(

                        hidden[i]

                        -

                        hidden[j]

                    )

                )

        return np.mean(dist)

    ####################################################################
    # Shuffle Hidden
    ####################################################################

    def shuffle_hidden(
        self,
        hidden,
    ):

        shuffled = hidden.copy()

        idx = np.arange(

            shuffled.shape[0]

        )

        np.random.shuffle(idx)

        return shuffled[idx]

    ####################################################################
    # Pair Analysis
    ####################################################################

    def analyse_pair(

        self,

        cache_a,

        cache_b,

        prompt_a,

        prompt_b,

    ):

        hidden_a = self.load_layer_matrix(

            cache_a

        )

        hidden_b = self.load_layer_matrix(

            cache_b

        )

        cosine_layer = self.cosine_curve(

            hidden_a,

            hidden_b

        )

        cka_layer = self.cka_curve(

            hidden_a,

            hidden_b

        )

        drift_a = self.drift_curve(

            hidden_a

        )

        drift_b = self.drift_curve(

            hidden_b

        )

        collapse_a = self.collapse_score(

            hidden_a

        )

        collapse_b = self.collapse_score(

            hidden_b

        )

        shuffle = self.shuffle_hidden(

            hidden_b

        )

        shuffle_cosine = self.cosine_curve(

            hidden_a,

            shuffle

        )

        return {

            "prompt_a": prompt_a,

            "prompt_b": prompt_b,

            "cosine": cosine_layer,

            "cka": cka_layer,

            "shuffle": shuffle_cosine,

            "collapse_a": collapse_a,

            "collapse_b": collapse_b,

            "drift_a": drift_a,

            "drift_b": drift_b,

        }

    ####################################################################
    # Cross Query Baseline
    ####################################################################

    def random_cross_query(

        self,

        prompt_name,

    ):

        ids = list(

            range(

                len(self.index)

            )

        )

        random.shuffle(ids)

        cosine_values = []

        cka_values = []

        for i in range(

            len(ids) - 1

        ):

            row_a = self.index.iloc[

                ids[i]

            ]

            row_b = self.index.iloc[

                ids[i + 1]

            ]

            hidden_a = self.load_layer_matrix(

                row_a[prompt_name]

            )

            hidden_b = self.load_layer_matrix(

                row_b[prompt_name]

            )

            cosine_values.append(

                self.cosine_curve(

                    hidden_a,

                    hidden_b

                )

            )

            cka_values.append(

                self.cka_curve(

                    hidden_a,

                    hidden_b

                )

            )

        cosine_values = np.stack(

            cosine_values

        )

        cka_values = np.stack(

            cka_values

        )

        return {

            "cosine_mean":

                cosine_values.mean(

                    axis=0

                ),

            "cka_mean":

                cka_values.mean(

                    axis=0

                ),

            "cosine_std":

                cosine_values.std(

                    axis=0

                ),

            "cka_std":

                cka_values.std(

                    axis=0

                )

        }

    ####################################################################
    # Pairwise Prompt Analysis
    ####################################################################

    def run_pairs(self):

        for _, row in tqdm(

            self.index.iterrows(),

            total=len(self.index)

        ):

            for i in range(

                len(

                    self.prompt_names

                )

            ):

                for j in range(

                    i + 1,

                    len(

                        self.prompt_names

                    )

                ):

                    pa = self.prompt_names[i]

                    pb = self.prompt_names[j]

                    result = self.analyse_pair(

                        row[pa],

                        row[pb],

                        pa,

                        pb,

                    )

                    result["sample"] = row["id"]

                    self.results.append(

                        result

                    )
    ####################################################################
    # Aggregate Statistics
    ####################################################################

    def aggregate(self):

        cosine = []
        cka = []
        shuffle = []

        drift_a = []
        drift_b = []

        collapse_a = []
        collapse_b = []

        for item in self.results:

            cosine.append(item["cosine"])
            cka.append(item["cka"])
            shuffle.append(item["shuffle"])

            drift_a.append(item["drift_a"])
            drift_b.append(item["drift_b"])

            collapse_a.append(item["collapse_a"])
            collapse_b.append(item["collapse_b"])

        cosine = np.stack(cosine)
        cka = np.stack(cka)
        shuffle = np.stack(shuffle)

        drift_a = np.stack(drift_a)
        drift_b = np.stack(drift_b)

        summary = {

            "cosine_mean":
                cosine.mean(axis=0),

            "cosine_std":
                cosine.std(axis=0),

            "cka_mean":
                cka.mean(axis=0),

            "cka_std":
                cka.std(axis=0),

            "shuffle_mean":
                shuffle.mean(axis=0),

            "shuffle_std":
                shuffle.std(axis=0),

            "drift_mean":
                np.concatenate(
                    [drift_a, drift_b],
                    axis=0
                ).mean(axis=0),

            "drift_std":
                np.concatenate(
                    [drift_a, drift_b],
                    axis=0
                ).std(axis=0),

            "collapse_mean":
                np.mean(
                    collapse_a + collapse_b
                ),

            "collapse_std":
                np.std(
                    collapse_a + collapse_b
                ),

        }

        return summary

    ####################################################################
    # Save Pairwise Results
    ####################################################################

    def save_pairwise(self):

        rows = []

        for item in self.results:

            rows.append({

                "sample":
                    item["sample"],

                "prompt_a":
                    item["prompt_a"],

                "prompt_b":
                    item["prompt_b"],

                "collapse_a":
                    item["collapse_a"],

                "collapse_b":
                    item["collapse_b"],

                "cosine_last":
                    item["cosine"][-1],

                "cka_last":
                    item["cka"][-1],

                "shuffle_last":
                    item["shuffle"][-1],

            })

        df = pd.DataFrame(rows)

        out = os.path.join(

            RESULT_DIR,

            "geometry_pairwise.csv"

        )

        df.to_csv(

            out,

            index=False

        )

        return df

    ####################################################################
    # Save Layer Curves
    ####################################################################

    def save_layer_statistics(self):

        summary = self.aggregate()

        layer_num = len(

            summary["cosine_mean"]

        )

        df = pd.DataFrame({

            "layer":

                np.arange(layer_num),

            "cosine_mean":

                summary["cosine_mean"],

            "cosine_std":

                summary["cosine_std"],

            "cka_mean":

                summary["cka_mean"],

            "cka_std":

                summary["cka_std"],

            "shuffle_mean":

                summary["shuffle_mean"],

            "shuffle_std":

                summary["shuffle_std"]

        })

        df.to_csv(

            os.path.join(

                RESULT_DIR,

                "geometry_layers.csv"

            ),

            index=False

        )

        return df

    ####################################################################
    # Cross-query Baseline
    ####################################################################

    def save_cross_query(self):

        rows = []

        for prompt in self.prompt_names:

            baseline = self.random_cross_query(

                prompt

            )

            for layer in range(

                len(

                    baseline["cosine_mean"]

                )

            ):

                rows.append({

                    "prompt":

                        prompt,

                    "layer":

                        layer,

                    "cosine":

                        baseline["cosine_mean"][layer],

                    "cosine_std":

                        baseline["cosine_std"][layer],

                    "cka":

                        baseline["cka_mean"][layer],

                    "cka_std":

                        baseline["cka_std"][layer]

                })

        df = pd.DataFrame(rows)

        df.to_csv(

            os.path.join(

                RESULT_DIR,

                "cross_query.csv"

            ),

            index=False

        )

        return df

    ####################################################################
    # Representation Distance Matrix
    ####################################################################

    def distance_matrix(

        self,

        sample_index,

        prompt,

    ):

        row = self.index.iloc[

            sample_index

        ]

        hidden = self.load_layer_matrix(

            row[prompt]

        )

        matrix = np.zeros(

            (

                hidden.shape[0],

                hidden.shape[0]

            )

        )

        for i in range(

            hidden.shape[0]

        ):

            for j in range(

                hidden.shape[0]

            ):

                matrix[i, j] = np.linalg.norm(

                    hidden[i]

                    -

                    hidden[j]

                )

        np.save(

            os.path.join(

                RESULT_DIR,

                f"distance_matrix_{sample_index}_{prompt}.npy"

            ),

            matrix

        )

        return matrix

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        print("=" * 80)
        print("Representation Geometry")
        print("=" * 80)

        self.run_pairs()

        pairwise = self.save_pairwise()

        layers = self.save_layer_statistics()

        cross = self.save_cross_query()

        print()
        print(pairwise.head())
        print()
        print(layers.head())
        print()
        print(cross.head())

        return {

            "pairwise": pairwise,

            "layers": layers,

            "cross_query": cross,

        }


########################################################################
# Entry
########################################################################

def run_geometry():

    experiment = GeometryExperiment()

    return experiment.run()


if __name__ == "__main__":

    run_geometry()
