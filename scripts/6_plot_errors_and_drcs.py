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
    group = True
    plot_errors = False
    plot_DRC_bars = True
    plot_DRC_markers = False
    string_TP = "300C_2MPa" # "300C_2MPa" | "340C_4MPa" | "380C_6MPa"

    # Get database names.
    ensemble = False
    if model_key == "DFT":
        task_dir = "reference"
        model_key_ts = "DFT"
    else:
        task_dir = "groupval" if group is True else "crossval"
        model_key_ts = f"{model_key}_from_{model_key}"
    db_ts_name = f"databases/{task_dir}/ZrO2_reactions_{model_key_ts}.db"

    # List of surfaces.
    surface_list = yaml.safe_load(open("yaml/surfaces.yaml", "r"))
    # Load results.
    db_model = connect(db_ts_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_model)

    # Plot parameters.
    plot_parameters = yaml.safe_load(open("yaml/plot_parameters.yaml", "r"))
    color_dict = plot_parameters["colors"]
    label_dict = plot_parameters["labels"]
    marker_dict = plot_parameters["markers"]
    species_key_dict = plot_parameters["species_key"]
    names_list, species_list = zip(*species_key_dict.items())

    # Plot DRCs.
    filename = f"results_kinetics/{task_dir}/results_{model_key}.yaml"
    with open(filename, "r") as fileobj:
        results_dict = yaml.safe_load(fileobj)
    fig, ax = plt.subplots(figsize=(4, 4))
    x_base = np.arange(len(names_list))
    width = 0.8 / len(surface_list)
    for results in results_dict.values():
        surface = results["surface"]
        temperature_C = results["temperature_C"]
        pressure_MPa = results["pressure_MPa"]
        marker = marker_dict[f"{temperature_C} C, {pressure_MPa} MPa"]
        color = color_dict[surface]
        jj = surface_list.index(surface)
        if plot_DRC_bars is True:
            for ii, species in enumerate(species_list):
                if f"{temperature_C:.0f}C_{pressure_MPa:.0f}MPa" != string_TP:
                    continue
                ax.bar(
                    x=x_base[ii] + (jj - (len(surface_list) - 1) / 2) * width,
                    height=results["DRC_dict"][species],
                    color=color,
                    width=width,
                    label=label_dict[surface],
                    edgecolor="black",
                )
        if plot_DRC_markers is True:
            # Plot markers for all conditions.
            for ii, species in enumerate(species_list):
                ax.plot(
                    x_base[ii] + (jj - (len(surface_list) - 1) / 2) * width,
                    results["DRC_dict"][species],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=8,
                    markeredgecolor="black",
                )
    # Add vertical lines.
    for xi in x_base[:-1]:
        ax.axvline(xi + 0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Degree of rate control [-]")
    ax.set_ylim(0, 1)
    # Add boxes with the names.
    ax.set_xlim(-0.5, len(names_list) - 0.5)
    ax.get_xaxis().set_visible(False)
    table = plt.table(
        cellText=[names_list],
        loc="bottom",
        cellLoc="center",
    )
    table.scale(1, 2.5)
    # Format y ticks.
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%+.2f$"))
    # Save figure.
    fig.tight_layout()
    filename = f"results_kinetics/{task_dir}/drc_{model_key}_{string_TP}.png"
    plt.savefig(filename, dpi=300)

    # Plot energy differences.
    if model_key_ts == "DFT" or plot_errors is False:
        return
    delta_energy_dict = {}
    for name, species in species_key_dict.items():
        delta_energy_dict[name] = {}
        for surface in surface_list:
            atoms = [
                atoms for atoms in atoms_list if atoms.info["surface"] == surface
                and atoms.info["species"] == species
            ][0]
            delta_energy = atoms.info["E_form"] - atoms.info["E_form_DFT"]
            delta_energy_dict[name][surface] = delta_energy
    # Plot bars.
    fig, ax = plt.subplots(figsize=(4, 4))
    names = list(delta_energy_dict.keys())
    x_base = np.arange(len(names))
    width = 0.8 / len(surface_list)
    for ii, surface in enumerate(surface_list):
        ax.bar(
            x=x_base + (ii - (len(surface_list) - 1) / 2) * width,
            height=[delta_energy_dict[name][surface] for name in names],
            width=width,
            color=color_dict[surface],
            label=label_dict[surface],
            edgecolor="black",
        )
    # Add horizontal and vertical lines.
    ax.axhline(0.0, color="black", linewidth=1)
    for xi in x_base[:-1]:
        ax.axvline(x=xi + 0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-0.5, +0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Energy difference [eV]")
    ax.legend(edgecolor="black", framealpha=1)
    # Add boxes with the names.
    ax.set_xlim(-0.5, len(names_list) - 0.5)
    ax.get_xaxis().set_visible(False)
    table = plt.table(
        cellText=[names_list],
        loc="bottom",
        cellLoc="center",
    )
    table.scale(1, 2.5)
    # Format y ticks.
    ax.set_yticks([-0.50, -0.25, +0.00, +0.25, +0.50])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r"$%+.2f$"))
    # Save figure.
    fig.tight_layout()
    plt.savefig(f"results_kinetics/{task_dir}/errors_react_{model_key}.png", dpi=300)

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