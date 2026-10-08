# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ase.db import connect

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.workflow import (
    update_ts_atoms,
    get_atoms_ref,
    get_crossvalidator,
    crossvalidation,
    calibrate_uncertainty,
    parity_plot,
    groups_errors_plot,
    uncertainty_plot,
)
from ase_ml_models.utilities import modify_name

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Control.
    scan_parameter = False
    # Run cross-validations.
    if scan_parameter is True:
        scan_model_parameter()
    else:
        run_crossvalidation()

# -------------------------------------------------------------------------------------
# RUN CROSS-VALIDATION
# -------------------------------------------------------------------------------------

def run_crossvalidation(
    param: str = None,
    param_value: float = None,
) -> dict:
    """
    Run cross-validation.
    """
    # Cross-validation parameters.
    species_type = "adsorbates" # adsorbates | reactions # Type of species.
    stratified = True # Stratified cross-validation.
    group = True # Group cross-validation.
    ensemble = False # Use the cross-validator to get an ensemble of models.
    key_groups = "surface" # surface | elements # Key for grouping the data.
    key_stratify = "species" # Key for stratification.
    n_splits = 5 # Number of splits for cross-validation.
    random_state = 42 # Random state for reproducibility.
    store_data = True # Store the data in an ASE database.
    add_ref_atoms = True # Add reference atoms to the training set.
    exclude_add = True # Exclude the reference atoms in the errors evaluation.
    print_error_thr = None # Threshold for printing the errors of the predictions.
    
    # Model selection.
    model_key = "Graph" # Linear | Graph
    update_features = False # Update features of TS atoms from an ASE database.
    model_key_ref = model_key[:] # Linear | Graph # Reference for updating features.
    
    # Linear models names.
    linear_dict = {"adsorbates": "TSR", "reactions": "BEP"}
    model_name = linear_dict[species_type] if model_key == "Linear" else model_key
    model_name_ref = "TSR" if model_key_ref == "Linear" else model_key_ref

    # Update model key if updating features from a reference model.
    if species_type == "reactions" and update_features is True:
        model_key = f"{model_key}_from_{model_key_ref}"

    # Task directory name.
    task_dir = "groupval" if group is True else "crossval"
    task_dir += "_ens" if ensemble is True else ""
    os.makedirs(f"databases/{task_dir}", exist_ok=True)

    # Model parameters.
    species_ref = ["H(Ga)+H(O)", "OH(Zr)+H(O)", "H2O(Ga)", "HCOO(Zr,Zr)+H(O)"]
    # Get model parameters.
    model_params = get_model_parameters(model_name=model_name)
    
    # Read ASE database.
    db_ase_name = f"databases/reference/ZrO2_{species_type}_DFT.db"
    db_ase = connect(db_ase_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_ase)
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        atoms_add = get_atoms_ref(atoms_list=atoms_list, species_ref=species_ref)
    else:
        atoms_add = []
    
    # Update TS features from an ASE database.
    if species_type == "reactions":
        task_dir_ref = task_dir if update_features is True else "reference"
        model_key_ref = model_key_ref if update_features is True else "DFT"
        db_ads_name = f"databases/{task_dir_ref}/ZrO2_adsorbates_{model_key_ref}.db"
        db_ads = connect(db_ads_name)
        e_form_dict = {"H2": +0.00, "CH2O": -0.40}
        filename_yaml = "yaml/reactants_products.yaml"
        reactants_products_dict = yaml.safe_load(open(filename_yaml, "r"))
        update_ts_atoms(
            atoms_list=atoms_list,
            db_ads=db_ads,
            e_form_dict=e_form_dict,
            reactants_products_dict=reactants_products_dict,
            features_key_dict={
                "E_first": "energy_first",
                "E_last": "energy_last",
                "ΔE_react": "energy_react",
            },
        )
    
    # Preprocess the data.
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_prepare
        fixed_TSR = model_params.pop("fixed_TSR", {})
        tsr_prepare(atoms_list, species_TSR=species_ref, fixed_TSR=fixed_TSR)
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list)
    elif model_name == "Graph":
        from ase_ml_models.graph import graph_preprocess, precompute_distances
        node_weight_dict = {
            "A0": 1.00,
            "S1": 0.90,
            "S2": 0.10,
        }
        edge_weight_dict = {
            "AA": 1.00,
            "AS": 1.00,
            "SS": 1.00,
        }
        feature_weight_dict = {
            "const": 1,
            "bader": 1,
            "soap": 1,
            "pdos": 1,
            "orbitals": 1,
            "energy": 10,
        }
        # Update model parameters if scanning a parameter.
        if param in node_weight_dict:
            node_weight_dict[param] = param_value
        elif param in edge_weight_dict:
            edge_weight_dict[param] = param_value
        elif param in feature_weight_dict:
            feature_weight_dict[param] = param_value
        elif param == "length_scale":
            del model_params["kwargs_kernel"]["length_scale_bounds"]
            model_params["kwargs_kernel"]["length_scale"] = 10 ** param_value
        elif param == "alpha":
            model_params["kwargs_model"]["alpha"] = 10 ** param_value
        # Preprocess graph data and precompute distances.
        graph_preprocess(
            atoms_list=atoms_list,
            node_weight_dict=node_weight_dict,
            edge_weight_dict=edge_weight_dict,
            feature_weight_dict=feature_weight_dict,
            stack_features=False,
            weight_neigh=0.10,
            n_iter=1,
            nan=-1,
        )
        filename = None
        distances = precompute_distances(atoms_X=atoms_list, filename=filename)
        model_params.update({"distances": distances})
    
    # Print number of data.
    print(f"n data: {len(atoms_list)}")
    print(f"n added: {len(atoms_add)}")
    
    # Initialize cross-validation.
    crossval = get_crossvalidator(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Prepare ASE database.
    db_model_name = f"databases/{task_dir}/ZrO2_{species_type}_{model_key}.db"
    db_model = connect(db_model_name, append=False) if store_data else None
    db_kwargs = {"keys_store": ["name", "surface", "species"]}
    # Cross-validation.
    results = crossvalidation(
        atoms_list=atoms_list,
        model_name=model_name,
        crossval=crossval,
        key_groups=key_groups,
        key_stratify=key_stratify,
        atoms_add=atoms_add,
        exclude_add=exclude_add,
        db_model=db_model,
        model_params=model_params,
        ensemble=ensemble,
        print_error_thr=print_error_thr,
        db_kwargs=db_kwargs,
    )
    
    # Get colors for plots.
    plot_parameters = yaml.safe_load(open("yaml/plot_parameters.yaml", "r"))
    color_dict = plot_parameters["colors"]
    surfaces = [atoms_list[ii].info["surface"] for ii in results["indices"]]
    color_list = [color_dict[surface] for surface in surfaces]

    # Plots parameters.
    plot_parity = True # Parity plot of predicted energies vs DFT energies.
    plot_species = False # Violin plots of errors distinguished by species.
    plot_surface = False # Violin plots of errors distinguished by surface.
    plot_uncertainty = False # Parity plot of uncertainty vs error.
    replace_dict = {**plot_parameters["replace_dict"], "<=>": "⇌"}
    results_dir = f"results_models/{task_dir}"
    os.makedirs(results_dir, exist_ok=True)
    # Parity plot.
    if plot_parity is True:
        lims = [-3.6, +1.0]
        ax = parity_plot(results=results, lims=lims, color=color_list, alpha=1.0)
        plt.savefig(f"{results_dir}/parity_{species_type}_{model_key}.png")
    # Species error plot.
    if plot_species is True:
        kwargs_name = {"replace_dict": replace_dict, "subscript_numbers": True}
        kwargs = {"key": "species", "color": "purple", "kwargs_name": kwargs_name}
        ax = groups_errors_plot(results=results, atoms_list=atoms_list, **kwargs)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/species_{species_type}_{model_key}.png")
    # Surfaces error plot.
    if plot_surface is True:
        kwargs_name = {"replace_dict": replace_dict, "subscript_numbers": False}
        kwargs = {"key": "surface", "color": color_dict, "kwargs_name": kwargs_name}
        kwargs["kwargs_xticks"] = {"rotation": 0., "ha": "center"}
        ax = groups_errors_plot(results=results, atoms_list=atoms_list, **kwargs)
        ax.tick_params(axis="x", labelsize=16, pad=10)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/surface_{species_type}_{model_key}.png")
    # Uncertainty quantification.
    if plot_uncertainty is True and "y_std" in results:
        results = calibrate_uncertainty(results=results, fit_intercept=False)
        ax = uncertainty_plot(results=results, color="grey")
        plt.savefig(f"{results_dir}/uncertainty_{species_type}_{model_key}.png")
    
    # Return results.
    return results

# -------------------------------------------------------------------------------------
# GET MODEL PARAMETERS
# -------------------------------------------------------------------------------------

def get_model_parameters(
    model_name: str,
):
    """
    Get model parameters based on the model name and species type. 
    """
    # Set target energy based on species type.
    target = "E_form"
    # TSR parameters.
    if model_name == "TSR":
        model_params = {"keys_TSR": ["species"]}
    # BEP parameters.
    elif model_name == "BEP":
        model_params = {"keys_BEP": ["species"]}
    # SKLearn parameters.
    elif model_name == "SKLearn":
        model_sklearn = "RandomForest"
        if model_sklearn == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        elif model_sklearn == "GPR":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF
            kernel = ConstantKernel(constant_value=1.0) * RBF(length_scales=100)
            model = GaussianProcessRegressor(kernel=kernel)
        model_params = {"model": model, "target": target}
    # Graph (WWL-GPR) parameters.
    if model_name == "Graph":
        model_params = {
            "target": target,
            "model_name": "GPR",
            "kwargs_kernel": {"length_scale_bounds": (10, 1000)},
            "kwargs_model": {"alpha": 1e-6},
        }
    return model_params

# -------------------------------------------------------------------------------------
# SCAN MODEL PARAMETERS
# -------------------------------------------------------------------------------------

def scan_model_parameter():
    """
    Scan model parameters and plot results.
    """
    # Control.
    params_dict = {
        "const": [0, 1, 2, 5, 10, 20],
        "bader": [0, 1, 2, 5, 10, 20],
        "soap": [0, 1, 2, 5, 10, 20],
        "pdos": [0, 1, 2, 5, 10, 20],
        "orbitals": [0, 1, 2, 5, 10, 20],
        "energy": [0, 1, 2, 5, 10, 20],
        "A0": [0.50, 0.75, 1.00],
        "S1": [0.50, 0.75, 1.00],
        "S2": [0.00, 0.25, 0.50],
        "AA": [0.50, 0.75, 1.00],
        "AS": [0.50, 0.75, 1.00],
        "SS": [0.50, 0.75, 1.00],
        "alpha": [-12, -9, -6, -3],
        "length_scale": [0, 1, 2, 3, 4],
    }
    labels_dict = {
        "const": "Weight physical constants features",
        "bader": "Weight Bader charges features",
        "soap": "Weight SOAP features",
        "pdos": "Weight pDOS features",
        "orbitals": "Weight HOMO & LUMO features",
        "energy": "Weight energy features",
        "A0": "Weight adsorbate nodes",
        "S1": "Weight first surface shell nodes",
        "S2": "Weight second surface shell nodes",
        "AA": "Weight adsorbate-adsorbate edges",
        "AS": "Weight adsorbate-surface edges",
        "SS": "Weight surface-surface edges",
        "length_scale": "log$_{10}$(length scale)",
        "alpha": "log$_{10}$(regularization)",
    }
    metric = "MAE"
    # Run cross-validations and plot results.
    for param_name, param_values in params_dict.items():
        print(f"Scanning parameter: {param_name}")
        mae_list = []
        for param_value in param_values:
            results = run_crossvalidation(param=param_name, param_value=param_value)
            mae_list.append(results[metric])
        # Plot results.
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
        ax.plot(param_values, mae_list, marker="o", color="crimson")
        ax.set_xlabel(labels_dict[param_name])
        ax.set_ylabel(f"{metric} [eV]")
        ax.set_ylim(0.0, 0.2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        plt.tight_layout()
        # Save figure.
        os.makedirs("results_scan", exist_ok=True)
        plt.savefig(f"results_scan/scan_{param_name}_{metric}.png")

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