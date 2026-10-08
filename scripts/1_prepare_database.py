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

from ase_ml_models.databases import (
    get_atoms_from_db,
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from ase_ml_models.utilities import get_connectivity, plot_connectivity
from ase_ml_models.features import (
    get_features_const,
    get_features_soap,
    write_features,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    for species_type in ["adsorbates", "reactions"]:
        prepare_database(species_type=species_type)

# -------------------------------------------------------------------------------------
# PREPARE DATABASE
# -------------------------------------------------------------------------------------

def prepare_database(
    species_type: str = "adsorbates",
):

    # Parameters.
    db_new_name = f"databases/reference/ZrO2_{species_type}_DFT.db"
    show_atoms = False
    write_db = True
    write_out = False
    filename_out = f"features_{species_type}.txt"
    # Features parameters.
    use_pdos_features = True
    use_homolumo_features = True
    # Constant features parameters.
    features_const_names = ["Eaff", "Eneg", "Ipot"]
    const_kwargs = {"features_names": features_const_names}
    # Bader charges.
    bader_charges_dict = yaml.safe_load(open("yaml/bader_charges.yaml", "r"))
    # SOAP parameters.
    n_max, l_max = 3, 2
    soap_kwargs = {"r_cut": 5.0, "sigma": 0.5, "n_max": n_max, "l_max": l_max}

    # Read atoms from database.
    db_all = connect(name="databases/reference/ZrO2_source_DFT.db")
    # Get all atoms.
    if species_type == "adsorbates":
        db_kwargs = {"kind": "adsorbates", "relaxed": True}
        atoms_list = get_atoms_list_from_db(db_ase=db_all, **db_kwargs)
    elif species_type == "reactions":
        db_kwargs = {"kind": "reactions", "relaxed": True, "image": "TS"}
        atoms_list = get_atoms_list_from_db(db_ase=db_all, **db_kwargs)

    # List of surfaces.
    surface_list = yaml.safe_load(open("yaml/surfaces.yaml", "r"))
    # Species dictionary.
    species_names = yaml.safe_load(open("yaml/species_names.yaml", "r"))
    species_dict = {**species_names["adsorbates"], **species_names["reactions"]}
    # Gas corrections.
    energy_correction_dict = yaml.safe_load(open("yaml/gas_corrections.yaml", "r"))

    # PDOS features.
    pdos_features_dict = yaml.safe_load(open("yaml/pdos_features.yaml", "r"))
    pdos_names = pdos_features_dict.pop("features_names")

    # HOMO-LUMO features.
    homolumo_features_dict = yaml.safe_load(open("yaml/homo_lumo.yaml", "r"))
    homolumo_names = homolumo_features_dict.pop("features_names")

    # Get reference energy of atomic species.
    gas_species_ref = ["H2", "H2O", "CO2"]
    energy_ref_funs = {
        "H" : lambda energy: energy["H2"] / 2,
        "O" : lambda energy: energy["H2O"] - 2 * energy["H"],
        "C" : lambda energy: energy["CO2"] - 2 * energy["O"],
    }
    energy_ref_dict = {}
    for species in gas_species_ref:
        gas_kwargs = {"kind": "molecules", "species": species, "relaxed": True}
        atoms = get_atoms_from_db(db_ase=db_all, **gas_kwargs)
        energy_corr = energy_correction_dict.get(species, 0.0)
        energy_ref_dict[species] = atoms.get_potential_energy() + energy_corr
    for name in energy_ref_funs:
        energy_ref_dict[name] = energy_ref_funs[name](energy=energy_ref_dict)

    # Show atoms.
    if show_atoms:
        gui = GUI(atoms_list)
        gui.run()

    # Calculate number features.
    n_const = len(features_const_names)
    n_soap = int(n_max * (n_max + 1) / 2 * (l_max + 1))
    # Features names.
    features_names = [f"const_{feat}" for feat in features_const_names]
    features_names += ["bader_chg"]
    features_names += [f"soap_{ii + 1:02d}" for ii in range(n_soap)]
    if use_pdos_features is True:
        features_names += pdos_names
    if use_homolumo_features is True:
        features_names += homolumo_names
    if species_type == "reactions":
        features_names += ["energy_first", "energy_last", "energy_react"]
    # SOAP dictionary.
    soap_dict = {}
    const_kwargs.update({"features_dict": {}})
    # Read atoms and get features.
    atoms_new_list = []
    for atoms in atoms_list:
        # Get surface and species from the database info.
        surface = atoms.info["surface"]
        species = atoms.info["species"]
        if surface not in surface_list or species not in species_dict:
            continue
        # Get new name of the species.
        species_new = species_dict[species]
        if species_type == "adsorbates":
            species_gas = atoms.info["species"]
        elif species_type == "reactions":
            species_gas = atoms.info["species"]
        # Get clean surface atoms.
        clean_kwargs = {"species": "00_clean", "surface": surface, "relaxed": True}
        atoms_clean = get_atoms_from_db(db_ase=db_all, **clean_kwargs)
        n_clean = len(atoms_clean)
        # Get indices of adsorbate atoms.
        indices_ads = list(range(n_clean, len(atoms)))
        atoms.info["indices_ads"] = indices_ads
        # Calculate adsorbate composition.
        atoms_ads = atoms[indices_ads]
        composition = dict(Counter(atoms_ads.get_chemical_symbols()))
        # Calculate formation energy.
        e_ref = atoms_clean.get_potential_energy()
        e_ref += sum(composition[ee] * energy_ref_dict[ee] for ee in composition)
        atoms.info["E_form"] = atoms.get_potential_energy() - e_ref
        atoms.info["E_ref"] = e_ref
        # Get connectivity.
        kwargs = {"method": "ase", "ensure_bonding": True, "skin": 0.2}
        if species_type == "adsorbates":
            atoms.info["connectivity"] = get_connectivity(atoms=atoms, **kwargs)
        elif species_type == "reactions":
            if get_info_reaction(atoms=atoms, db_ase=db_all, **kwargs) is False:
                continue
        # Get constants features.
        features_const = get_features_const(atoms=atoms, **const_kwargs)
        # Get SOAP features.
        if surface not in soap_dict:
            soap_dict[surface] = get_features_soap(atoms=atoms_clean, **soap_kwargs)
        if species_gas not in soap_dict:
            soap_dict[species_gas] = get_features_soap(atoms=atoms_ads, **soap_kwargs)
        features_soap = np.zeros((len(atoms), n_soap))
        features_soap[:n_clean] = soap_dict[surface]
        features_soap[n_clean:] = soap_dict[species_gas]
        # Get Bader charges.
        bader_charges = np.array(bader_charges_dict[surface]) - 1.3
        bader_charges[bader_charges < 0] = 0.0
        features_bader = np.zeros(len(atoms))
        features_bader[:n_clean] = bader_charges
        features_bader = features_bader.reshape(-1, 1)
        # Assemble features matrix.
        features = np.hstack([features_const, features_bader, features_soap])
        # PDOS features.
        if use_pdos_features is True:
            pdos_features_clean = np.array(pdos_features_dict[surface])
            pdos_features = np.zeros((len(atoms), pdos_features_clean.shape[1]))
            pdos_features[:n_clean] = pdos_features_clean
            features = np.hstack([features, pdos_features])
        # HOMO-LUMO features.
        if use_homolumo_features is True:
            homolumo_features_species = np.array(homolumo_features_dict[species])
            homolumo_features = np.zeros((len(atoms), 2))
            homolumo_features[n_clean:] = homolumo_features_species
            features = np.hstack([features, homolumo_features])
        # Add energies of initial and final states.
        if species_type == "reactions":
            e_first = np.array([atoms.info["E_first"]] * len(atoms)).reshape(-1, 1)
            e_last = np.array([atoms.info["E_last"]] * len(atoms)).reshape(-1, 1)
            delta_e = np.array([atoms.info["ΔE_react"]] * len(atoms)).reshape(-1, 1)
            features = np.hstack([features, e_first, e_last, delta_e])
        # Store features in info.
        atoms.info["features"] = features
        atoms.info["features_names"] = features_names
        # Global features.
        atoms.info["features_ave"] = np.nanmean(features, axis=0)
        atoms.info["features_ave_names"] = features_names
        # Get name of the structure.
        name = "_".join([surface, species_new]).replace(" ", "_")
        atoms.info["name"] = name
        atoms.info["species"] = species_new
        # Print name and formation energy of the structure.
        e_form = atoms.info["E_form"]
        print(f"{name:<100} E_form = {e_form:+7.3f} [eV]")
        # Append atoms to the list.
        atoms_new_list.append(atoms)
    
    # Write atoms to database.
    if write_db and db_new_name is not None:
        db_new = connect(name=db_new_name, append=False)
        write_atoms_list_to_db(
            atoms_list=atoms_new_list,
            db_ase=db_new,
            keys_store=["name", "surface", "species"],
            keys_match=None,
        )
    # Write features to file.
    if write_out and filename_out is not None:
        write_features(atoms_list=atoms_new_list, filename=filename_out)

# -------------------------------------------------------------------------------------
# GET INFO REACTION
# -------------------------------------------------------------------------------------

def get_info_reaction(
    atoms: Atoms,
    db_ase: Database,
    **kwargs,
):
    """
    Get connectivity and energy of initial and final images.
    """
    db_kwargs = {key: atoms.info[key] for key in ["surface", "species"]}
    db_kwargs["relaxed"] = True
    atoms_IS = get_atoms_from_db(db_ase=db_ase, image="IS", **db_kwargs, none_ok=True)
    atoms_FS = get_atoms_from_db(db_ase=db_ase, image="FS", **db_kwargs, none_ok=True)
    if atoms_IS is None or atoms_FS is None:
        print(f"Missing IS or FS for {db_kwargs}")
        return False
    connectivity_IS = get_connectivity(atoms=atoms_IS, **kwargs)
    connectivity_FS = get_connectivity(atoms=atoms_FS, **kwargs)
    atoms.info["connectivity"] = connectivity_IS + connectivity_FS
    atoms.info["E_first"] = atoms_IS.get_potential_energy() - atoms.info["E_ref"]
    atoms.info["E_last"] = atoms_FS.get_potential_energy() - atoms.info["E_ref"]
    atoms.info["E_act"] = atoms.info["E_form"] - atoms.info["E_first"]
    atoms.info["ΔE_react"] = atoms.info["E_last"] - atoms.info["E_first"]
    return True

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