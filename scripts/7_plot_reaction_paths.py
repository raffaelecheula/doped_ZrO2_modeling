# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ase.db import connect

from ase_ml_models.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    model_key = "Linear" # DFT | Linear | Graph

    # Load RPA data.
    rpa_data = yaml.safe_load(open("results_kinetics/rpa/reaction_paths.yaml", "r"))
    surface_list = list(rpa_data.keys())
    species_list = list(rpa_data[surface_list[0]][model_key].keys())

    # Plot parameters.
    plot_parameters = yaml.safe_load(open("yaml/plot_parameters.yaml", "r"))
    color_dict = plot_parameters["colors"]
    label_dict = plot_parameters["labels"]
    marker_dict = plot_parameters["markers"]

    # Plot reaction paths.
    fig, ax = plt.subplots(figsize=(4, 4))
    x_base = np.arange(len(species_list))
    width = 0.8 / len(surface_list)
    for surface, results in rpa_data.items():
        color = color_dict[surface]
        jj = surface_list.index(surface)
        for ii, species in enumerate(species_list):
            ax.bar(
                x=x_base[ii] + (jj - (len(surface_list) - 1) / 2) * width,
                height=results[model_key][species],
                color=color,
                width=width,
                label=label_dict[surface],
                edgecolor="black",
            )
    # Add vertical lines.
    for xi in x_base[:-1]:
        ax.axvline(xi + 0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Pathway reaction contribution [%]")
    ax.set_ylim(0, 100)
    # Add boxes with the names.
    ax.set_xlim(-0.5, len(species_list) - 0.5)
    ax.get_xaxis().set_visible(False)
    names_list = [
        name.replace("2", r"$_2$").replace("3", r"$_3$") + "\npathway"
        for name in species_list
    ]
    table = plt.table(
        cellText=[names_list],
        loc="bottom",
        cellLoc="center",
    )
    table.scale(1, 2.5)
    # Format y ticks.
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%.0f$"))
    # Save figure.
    fig.tight_layout()
    plt.savefig(f"results_kinetics/rpa/reaction_paths_{model_key}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    print(f"Execution time: {timeit.default_timer() - start:.2f} [s]")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------