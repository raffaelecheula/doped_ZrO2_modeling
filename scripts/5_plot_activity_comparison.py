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
    model_key = "Graph" # Linear | Graph
    group = True
    ensemble = True

    # Get model names.
    task_dir = "groupval" if group is True else "crossval"
    task_dir += "_ens" if ensemble is True else ""
    model_key_ts = f"{model_key}_from_{model_key}"

    # Store results from DFT data in YAML file.
    with open(f"results_kinetics/{task_dir}/results_{model_key}.yaml", "r") as fileobj:
        results_dict = yaml.safe_load(fileobj)
    
    # Read results from DFT data from YAML file.
    with open("results_kinetics/reference/results_DFT.yaml", "r") as fileobj:
        results_DFT = yaml.safe_load(fileobj)
    # Update results dict with DFT results.
    for name in results_dict:
        results_dict[name]["yield_CH3OH_DFT"] = results_DFT[name]["yield_CH3OH"]

    # Plot parameters.
    plot_parameters = yaml.safe_load(open("yaml/plot_parameters.yaml", "r"))
    color_dict = plot_parameters["colors"]
    label_dict = plot_parameters["labels"]
    marker_dict = plot_parameters["markers"]
    # Plot results.
    y_max = 10.0
    fig, ax = plt.subplots(figsize=(4, 4))
    for name in results_dict:
        # Get results.
        surface = results_dict[name]["surface"]
        temperature_C = results_dict[name]["temperature_C"]
        pressure_MPa = results_dict[name]["pressure_MPa"]
        xx = results_dict[name]["yield_CH3OH_DFT"]
        yy = results_dict[name]["yield_CH3OH"]
        # Get mean and standard deviation.
        yerr = np.std(yy) if ensemble is True else None
        yy = np.mean(yy) if ensemble is True else yy
        # Plot results.
        ax.errorbar(
            x=xx,
            y=yy,
            yerr=yerr,
            fmt=marker_dict[f"{temperature_C} C, {pressure_MPa} MPa"],
            label=surface,
            color=color_dict[surface],
            markersize=8,
            markeredgecolor="black",
            capsize=5,
            ecolor="black",
        )
    ax.plot([0., y_max], [0., y_max], "k--", linewidth=1)
    ax.set_xlabel("DFT CH$_3$OH yield [%]")
    ax.set_ylabel("Model CH$_3$OH yield [%]")
    ax.set_xlim(0., y_max)
    ax.set_ylim(0., y_max)
    # Color legend.
    color_handles = []
    kwargs_surface = {"markersize": 8, "markeredgecolor": "black"}
    for surface, color in color_dict.items():
        label = label_dict[surface]
        handle, = ax.plot([], [], "o", color=color, label=label, **kwargs_surface)
        color_handles.append(handle)
    legend1 = ax.legend(handles=color_handles, loc="upper left", edgecolor="black")
    # Marker legend.
    marker_handles = []
    kwargs_marker = {"color": "grey", "linestyle": "None", **kwargs_surface}
    for label, marker in marker_dict.items():
        label = label.replace(" C", "°C")
        handle, = ax.plot([], [], marker, label=label, **kwargs_marker)
        marker_handles.append(handle)
    legend2 = ax.legend(handles=marker_handles, loc="lower right", edgecolor="black")
    ax.add_artist(legend1)
    # Save plot.
    fig.tight_layout()
    plt.savefig(f"results_kinetics/{task_dir}/activity_{model_key}.png", dpi=300)

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