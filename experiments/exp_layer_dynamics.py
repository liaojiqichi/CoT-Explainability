import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from extraction import load_signal
from metrics import cosine

########################################################################
# Experiment
#
# Layer Dynamics
#
# Fig10
# Representation Drift
#
# Fig11
# Layer Velocity
#
# Fig12
# Layer Acceleration
#
# Fig13
# Stable Layer Detection
########################################################################


class LayerDynamicsExperiment:

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
    # Drift
    ####################################################################

    def compute_drift(

        self,

        layer_matrix,

    ):

        drift = []

        for i in range(

            layer_matrix.shape[0]-1

        ):

            drift.append(

                np.linalg.norm(

                    layer_matrix[i+1]

                    -

                    layer_matrix[i]

                )

            )

        return np.asarray(drift)

    ####################################################################
    # Velocity
    ####################################################################

    def compute_velocity(

        self,

        drift,

    ):

        velocity = np.zeros_like(

            drift

        )

        velocity[1:] = (

            drift[1:]

            -

            drift[:-1]

        )

        return velocity

    ####################################################################
    # Acceleration
    ####################################################################

    def compute_acceleration(

        self,

        velocity,

    ):

        acc = np.zeros_like(

            velocity

        )

        acc[1:] = (

            velocity[1:]

            -

            velocity[:-1]

        )

        return acc

    ####################################################################
    # Stable Layer
    ####################################################################

    def stable_layer(

        self,

        drift,

    ):

        return int(

            np.argmin(

                drift

            )

        )

    ####################################################################
    # Transition Similarity
    ####################################################################

    def transition_similarity(

        self,

        layer_matrix,

    ):

        values = []

        for i in range(

            layer_matrix.shape[0]-1

        ):

            values.append(

                cosine(

                    layer_matrix[i],

                    layer_matrix[i+1]

                )

            )

        return 1 - np.asarray(values)

    ####################################################################
    # Layer Energy
    ####################################################################

    def layer_energy(

        self,

        layer_matrix,

    ):

        return np.linalg.norm(

            layer_matrix,

            axis=1

        )

    ####################################################################
    # Process One Sample
    ####################################################################

    def analyse_sample(

        self,

        cache_file,

        sample_id,

        prompt,

    ):

        signal = load_signal(

            cache_file

        )

        hidden = signal[

            "layer_matrix"

        ]

        drift = self.compute_drift(

            hidden

        )

        velocity = self.compute_velocity(

            drift

        )

        acceleration = self.compute_acceleration(

            velocity

        )

        transition = self.transition_similarity(

            hidden

        )

        energy = self.layer_energy(

            hidden

        )

        self.records.append(

            {

                "sample":

                    sample_id,

                "prompt":

                    prompt,

                "drift":

                    drift,

                "velocity":

                    velocity,

                "acceleration":

                    acceleration,

                "transition":

                    transition,

                "energy":

                    energy,

                "stable_layer":

                    self.stable_layer(

                        drift

                    )

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

                self.analyse_sample(

                    row[prompt],

                    sid,

                    prompt

                )

        return self.save_results()

    ####################################################################
    # Aggregate
    ####################################################################

    def aggregate(

        self,

        key,

    ):

        arr = np.stack(

            [

                x[key]

                for x in self.records

            ]

        )

        return (

            arr.mean(axis=0),

            arr.std(axis=0)

        )

    ####################################################################
    # Save
    ####################################################################

    def save_results(self):

        drift_mean, drift_std = self.aggregate(

            "drift"

        )

        velocity_mean, velocity_std = self.aggregate(

            "velocity"

        )

        acc_mean, acc_std = self.aggregate(

            "acceleration"

        )

        transition_mean, transition_std = self.aggregate(

            "transition"

        )

        energy_mean, energy_std = self.aggregate(

            "energy"

        )

        stable = [

            x["stable_layer"]

            for x in self.records

        ]

        df = pd.DataFrame(

            {

                "layer":

                    np.arange(

                        len(

                            drift_mean

                        )

                    ),

                "drift_mean":

                    drift_mean,

                "drift_std":

                    drift_std,

                "velocity_mean":

                    velocity_mean,

                "velocity_std":

                    velocity_std,

                "acceleration_mean":

                    acc_mean,

                "acceleration_std":

                    acc_std,

                "transition_mean":

                    transition_mean,

                "transition_std":

                    transition_std,

                "energy_mean":

                    energy_mean[:-1],

                "energy_std":

                    energy_std[:-1],

            }

        )

        df.to_csv(

            os.path.join(

                RESULT_DIR,

                "layer_dynamics.csv"

            ),

            index=False

        )

        stable_df = pd.DataFrame(

            {

                "stable_layer":

                    stable

            }

        )

        stable_df.to_csv(

            os.path.join(

                RESULT_DIR,

                "stable_layers.csv"

            ),

            index=False

        )

        return df


########################################################################
# Entry
########################################################################

def run_layer_dynamics():

    exp = LayerDynamicsExperiment()

    return exp.run()


if __name__ == "__main__":

    run_layer_dynamics()
