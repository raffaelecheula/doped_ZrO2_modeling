# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
from collections import Counter
from ase import Atoms
from ase.db.core import Database
from ase.io import read
from ase.db import connect
from ase.gui.gui import GUI
from ase.thermochemistry import HarmonicThermo

from ase_ml_models.databases import get_atoms_from_db, get_atoms_list_from_db
from ase_ml_models.yaml import write_to_yaml
from ase_cantera_microkinetics import units
from ase_cantera_microkinetics.thermochemistry import ase_thermo_to_NASA_coeffs
from ase_cantera_microkinetics.reaction_mechanism import NameAnalyzer

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Gas phase species.
    species_gas = yaml.safe_load(open("yaml/gas_thermodynamics.yaml", "r"))["species"]
    # Gas corrections.
    energy_correction_dict = yaml.safe_load(open("yaml/gas_corrections.yaml", "r"))
    # Surface to analyze.
    surface = "ZrO2(101)+Zn2+"
    free_sites = ["(Zr)", "(Ga)", "(O)"]
    # Species names.
    species_names = yaml.safe_load(open("yaml/species_names.yaml", "r"))

    # Prepare database.
    db_all = connect(name="databases/reference/ZrO2_source_DFT.db")
    # Get reference and zero-point energies of atomic species.
    gas_species_ref = ["H2", "H2O", "CO2"]
    energy_ref_funs = {
        "H": lambda energy: energy["H2"] / 2,
        "O": lambda energy: energy["H2O"] - 2 * energy["H"],
        "C": lambda energy: energy["CO2"] - 2 * energy["O"],
    }
    energy_ZP_dict = {}
    for species in gas_species_ref:
        # Get zero-point energies of the gas phase species.
        gas_kwargs = {"kind": "molecules", "species": species, "relaxed": True}
        atoms = get_atoms_from_db(db_ase=db_all, **gas_kwargs)
        thermo = HarmonicThermo(vib_energies=atoms.info["vib_energies"])
        energy_ZP_dict[species] = thermo.get_ZPE_correction()
    # Calculate ZP energies of atomic species.
    for name in energy_ref_funs:
        energy_ZP_dict[name] = energy_ref_funs[name](energy=energy_ZP_dict)

    # Change gas reference energies.
    energy_ref_funs = {**energy_ref_funs, "N": lambda energy: energy["N2"] / 2}
    species_gas = change_gas_reference_energies(
        species_gas=species_gas,
        energy_ref_funs=energy_ref_funs,
    )
    
    # Get NASA coefficients of adsorbates species.
    db_kwargs = {"kind": "adsorbates", "surface": surface, "relaxed": True}
    atoms_list = get_atoms_list_from_db(db_ase=db_all, **db_kwargs)
    clean_kwargs = {"species": "00_clean", "surface": surface, "relaxed": True}
    atoms_clean = get_atoms_from_db(db_ase=db_all, **clean_kwargs)
    coeffs_adsorbates_dict = get_NASA_coefficients_from_atoms_list(
        atoms_list=atoms_list,
        n_atoms_clean=len(atoms_clean),
        species_names_dict=species_names["adsorbates"],
        energy_ZP_dict=energy_ZP_dict,
    )
    # Merge all adsorbate dictionaries.
    coeffs_free_dict = {key: [0.] * 7 for key in free_sites}
    coeffs_adsorbates_dict = {**coeffs_free_dict, **coeffs_adsorbates_dict}
    # Add coefficients of undoped ZrO2(101).
    for key in ["CO2(Ga,Zr,Zr,O)", "H(Ga)+H(O)"]:
        key_new = key.replace("Ga", "Zr")
        coeffs_adsorbates_dict[key_new] = coeffs_adsorbates_dict[key].copy()

    # Get NASA coefficients of transition states.
    db_kwargs = {"kind": "reactions", "surface": surface, "relaxed": True}
    atoms_list = get_atoms_list_from_db(db_ase=db_all, image="TS", **db_kwargs)
    coeffs_reactions_dict = get_NASA_coefficients_from_atoms_list(
        atoms_list=atoms_list,
        n_atoms_clean=len(atoms_clean),
        species_names_dict=species_names["reactions"],
        energy_ZP_dict=energy_ZP_dict,
    )
    # Add coefficients of undoped ZrO2(101).
    for key in ["H2 + (Ga) + (O) <=> H(Ga)+H(O)"]:
        key_new = key.replace("Ga", "Zr")
        coeffs_reactions_dict[key_new] = coeffs_reactions_dict[key].copy()

    # Names analyzer.
    name_analyzer = NameAnalyzer()
    # Adsorbate species.
    species_adsorbates = get_species_list_with_NASA_thermo(
        coeffs_dict=coeffs_adsorbates_dict,
        name_analyzer=name_analyzer,
    )
    # Reactions species.
    species_reactions = get_species_list_with_NASA_thermo(
        coeffs_dict=coeffs_reactions_dict,
        name_analyzer=name_analyzer,
        sticking=False,
    )
    # Sticking reactions species.
    names_stick_dict = species_names["reactions-sticking"]
    coeffs_stick_dict = {name: [0.] * 7 for name in names_stick_dict.values()}
    species_sticking = get_species_list_with_NASA_thermo(
        coeffs_dict=coeffs_stick_dict,
        name_analyzer=name_analyzer,
        sticking=True,
    )
    # Write the data to a YAML file.
    data = {
        "species-gas": species_gas,
        "species-adsorbates": species_adsorbates,
        "species-reactions": species_reactions + species_sticking,
    }
    write_to_yaml(filename="yaml/mechanism.yaml", data=data)

    # Print success message.
    print("Mechanism YAML file created successfully.")

# -------------------------------------------------------------------------------------
# CHANGE GAS REFERENCE ENERGIES
# -------------------------------------------------------------------------------------

def change_gas_reference_energies(
    species_gas: list,
    energy_ref_funs: dict,
):
    """
    Get the reference energies of the gas phase species.
    """
    energy_exp = {}
    for species in species_gas:
        name = species["name"]
        energy_exp[name] = species["thermo"]["data"][0][5]
    for name in energy_ref_funs:
        energy_exp[name] = energy_ref_funs[name](energy=energy_exp)
    for species in species_gas:
        name = species["name"]
        composition = species["composition"]
        e_form = energy_exp[name]
        for elem in composition:
            e_form -= energy_exp[elem] * composition[elem]
        species["thermo"]["data"][1][5] += e_form - species["thermo"]["data"][0][5]
        species["thermo"]["data"][0][5] = e_form
        del species["transport"]
    # Return the updated species_gas list.
    return species_gas

# -------------------------------------------------------------------------------------
# GET NASA COEFFICIENTS FROM ATOMS LIST
# -------------------------------------------------------------------------------------

def get_NASA_coefficients_from_atoms_list(
    atoms_list: list,
    n_atoms_clean: int,
    species_names_dict: dict,
    energy_ZP_dict: dict,
    n_points: int = 1000,
    temp_low: int = 200,
    temp_max: int = 1000,
):
    """
    Get NASA coefficients of the species in the database.
    """
    coeffs_dict = {}
    for atoms in atoms_list:
        # Filter species without vibrational energies or already included.
        name = species_names_dict.get(atoms.info["species"], None)
        if name is None or "vib_energies" not in atoms.info or name in coeffs_dict:
            continue
        # Get NASA coefficients.
        thermo = HarmonicThermo(vib_energies=atoms.info["vib_energies"])
        coeffs_dict[name] = ase_thermo_to_NASA_coeffs(
            thermo=thermo,
            n_points=n_points,
            t_low=temp_low,
            t_max=temp_max,
            subtract_ZPE=False,
        )
        # Get composition.
        atoms_ads = atoms[n_atoms_clean:]
        composition = dict(Counter(atoms_ads.get_chemical_symbols()))
        # Subtract gas ZP energy.
        energy_ZP_ref = sum(composition[ee] * energy_ZP_dict[ee] for ee in composition)
        coeffs_dict[name][5] -= energy_ZP_ref * units.eV / units.molecule / units.Rgas
    # Return NASA coefficients.
    return coeffs_dict

# -------------------------------------------------------------------------------------
# GET SPECIES LIST WITH NASA THERMO
# -------------------------------------------------------------------------------------

def get_species_list_with_NASA_thermo(
    coeffs_dict: dict,
    name_analyzer: NameAnalyzer,
    temp_low: int = 200,
    temp_max: int = 1000,
    sticking: bool = None,
):
    """
    Get list of species with NASA thermo.
    """
    # Adsorbate species.
    species_list = []
    for name in coeffs_dict:
        composition, size = name_analyzer.get_composition_and_size(name=name)
        species = {
            "name": name,
            "composition": composition,
            "size": size,
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [temp_low, temp_max],
                "data": [[float(ii) for ii in coeffs_dict[name]]],
            },
        }
        if sticking is not None:
            species["thermo"]["sticking"] = sticking
        species_list.append(species)
    # Return species list.
    return species_list

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