import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

########################################################################
# Plot Style
########################################################################

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.fontsize"] = 11

SAVE_DIR = os.path.join(
    FIGURE_DIR,
    "layer_dynamics"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

########################################################################
# Load
########################################################################

def load():

    return pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "layer_dynamics.csv"

        )

    )

########################################################################
# Fig10
# Representation Drift
########################################################################

def plot_drift():

    df = load()

    plt.figure(figsize=(8,5))

    plt.plot(

        df.layer,

        df.drift_mean,

        linewidth=2,

        label="Mean Drift"

    )

    plt.fill_between(

        df.layer,

        df.drift_mean-df.drift_std,

        df.drift_mean+df.drift_std,

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Representation Drift")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig10_drift.pdf"

        )

    )

    plt.close()

########################################################################
# Fig11
# Velocity
########################################################################

def plot_velocity():

    df = load()

    plt.figure(figsize=(8,5))

    plt.plot(

        df.layer,

        df.velocity_mean,

        linewidth=2

    )

    plt.fill_between(

        df.layer,

        df.velocity_mean-df.velocity_std,

        df.velocity_mean+df.velocity_std,

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Velocity")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig11_velocity.pdf"

        )

    )

    plt.close()

########################################################################
# Fig12
# Acceleration
########################################################################

def plot_acceleration():

    df = load()

    plt.figure(figsize=(8,5))

    plt.plot(

        df.layer,

        df.acceleration_mean,

        linewidth=2

    )

    plt.fill_between(

        df.layer,

        df.acceleration_mean-df.acceleration_std,

        df.acceleration_mean+df.acceleration_std,

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Acceleration")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig12_acceleration.pdf"

        )

    )

    plt.close()

########################################################################
# Fig13
# Transition Similarity
########################################################################

def plot_transition():

    df = load()

    plt.figure(figsize=(8,5))

    plt.plot(

        df.layer,

        df.transition_mean,

        linewidth=2,

        label="Transition"

    )

    plt.fill_between(

        df.layer,

        df.transition_mean-df.transition_std,

        df.transition_mean+df.transition_std,

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("Cosine(layer i, layer i+1)")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig13_transition.pdf"

        )

    )

    plt.close()

########################################################################
# Fig14
# Layer Energy
########################################################################

def plot_energy():

    df = load()

    plt.figure(figsize=(8,5))

    plt.plot(

        df.layer,

        df.energy_mean,

        linewidth=2,

        label="Energy"

    )

    plt.fill_between(

        df.layer,

        df.energy_mean-df.energy_std,

        df.energy_mean+df.energy_std,

        alpha=.25

    )

    plt.xlabel("Layer")

    plt.ylabel("L2 Norm")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig14_energy.pdf"

        )

    )

    plt.close()

########################################################################
# Fig15
# Stable Layer Histogram
########################################################################

def plot_stable_layer():

    df = pd.read_csv(

        os.path.join(

            RESULT_DIR,

            "stable_layers.csv"

        )

    )

    plt.figure(figsize=(7,5))

    plt.hist(

        df.stable_layer,

        bins=np.arange(

            df.stable_layer.max()+2

        )-.5

    )

    plt.xlabel("Stable Layer")

    plt.ylabel("Frequency")

    plt.grid(alpha=.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVE_DIR,

            "fig15_stable_layer.pdf"

        )

    )

    plt.close()

########################################################################
# Run
########################################################################

def run_layer_dynamics_plot():

    plot_drift()

    plot_velocity()

    plot_acceleration()

    plot_transition()

    plot_energy()

    plot_stable_layer()

if __name__ == "__main__":

    run_layer_dynamics_plot()
