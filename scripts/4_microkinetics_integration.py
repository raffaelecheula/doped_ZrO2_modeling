# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

from ase_cantera_microkinetics import units
from ase_cantera_microkinetics.cantera_utilities import (
    get_Y_dict,
    advance_sim_to_steady_state,
    degree_rate_control,
    generalized_degree_rate_control,
    molar_balance_of_element,
    get_std_gibbs_dict,
    reaction_path_analysis,
)
from ase_cantera_microkinetics.ase_utilities import (
    get_mechanism_from_yaml,
    get_e_form_from_ase_atoms,
    get_names_list_from_yaml,
    get_atoms_list_from_db,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Parameters.
    model_key = "Graph" # DFT | Linear | Graph
    group = True
    ensemble = False
    yaml_file = "yaml/mechanism.yaml"
    calculate_DRC = False
    write_results_yaml = True
    # Temperature and pressure combinations.
    temperature_pressure_list = [(300, 2), (340, 4), (380, 6)]

    # Get database names.
    if model_key == "DFT":
        task_dir = "reference"
        model_key_ts = "DFT"
        ensemble = False
    else:
        task_dir = "groupval" if group is True else "crossval"
        task_dir += "_ens" if ensemble is True else ""
        model_key_ts = f"{model_key}_from_{model_key}"
    db_ads_name = f"databases/{task_dir}/ZrO2_adsorbates_{model_key}.db"
    db_ts_name = f"databases/{task_dir}/ZrO2_reactions_{model_key_ts}.db"
    
    # List of surfaces.
    surface_list = yaml.safe_load(open("yaml/surfaces.yaml", "r"))
    # Energy indices.
    e_index_list = [0, 1, 2, 3] if ensemble is True else [None]
    # Main loop.
    results_dict = {}
    for surface in surface_list:
        for temperature_C, pressure_MPa in temperature_pressure_list:
            for e_index in e_index_list:
                results = kinetics_integration(
                    db_ads_name=db_ads_name,
                    db_ts_name=db_ts_name,
                    yaml_file=yaml_file,
                    surface=surface,
                    temperature_C=temperature_C,
                    pressure_MPa=pressure_MPa,
                    e_index=e_index,
                    calculate_DRC=calculate_DRC,
                )
                yield_CH3OH = float(results["conversion_CO2"] * 100) # [%]
                DRC_dict = results["DRC_dict"]
                # Compose name.
                name = f"{surface}_T={temperature_C:.0f}C_P={pressure_MPa:.0f}MPa"
                # Store results in a dictionary.
                yield_CH3OH = [yield_CH3OH] if ensemble is True else yield_CH3OH
                if name not in results_dict:
                    results_dict[name] = {
                        "surface": surface,
                        "temperature_C": temperature_C,
                        "pressure_MPa": pressure_MPa,
                        "yield_CH3OH": yield_CH3OH,
                        "DRC_dict": DRC_dict,
                    }
                else:
                    results_dict[name]["yield_CH3OH"].extend(yield_CH3OH)

    # Store results from DFT data in YAML file.
    os.makedirs(f"results_kinetics/{task_dir}", exist_ok=True)
    if write_results_yaml is True:
        filename = f"results_kinetics/{task_dir}/results_{model_key}.yaml"
        with open(filename, "w") as fileobj:
            yaml.dump(data=results_dict, stream=fileobj, sort_keys=False)
    
# -------------------------------------------------------------------------------------
# KINETICS INTEGRATION
# -------------------------------------------------------------------------------------

def kinetics_integration(
    db_ads_name: str,
    db_ts_name: str,
    yaml_file: str,
    surface: str,
    temperature_C: float = 300, # [°C]
    pressure_MPa: float = 2, # [MPa]
    e_index: int = None,
    print_molfracs: bool = False,
    print_coverages: bool = False,
    calculate_DRC: bool = False,
    element_balance: bool = False,
):
    # Print the parameters of the simulation.
    print(f"\nSurface = {surface}")
    print(f"Temperature = {temperature_C:.0f} [°C]")
    print(f"Pressure = {pressure_MPa:.0f} [MPa]")
    # Set temperature and pressure of the simulation.
    temperature = units.Celsius_to_Kelvin(temperature=temperature_C) # [K]
    pressure = pressure_MPa * units.mega * units.Pa # [Pa]
    # Set molar fractions of gas species.
    gas_molfracs_inlet = {"CO2": 0.24, "H2": 0.72, "N2": 0.04} # [mol/mol]
    # Set the initial coverages of the free catalytic sites.
    cat_coverages_inlet = {"(Zr)": 8 / 18, "(Ga)": 1 / 18, "(O)": 9 / 18}
    # Set number of CSTR for the discretization of the PFR.
    n_cstr = 100
    # Set the catalyst parameters.
    cat_mass = 0.10 * units.gram # [kg]
    cat_mol_weight = 123.22 * units.gram / units.mole # [kg/kmol]
    cat_density = 6.10 * units.gram / units.centimeter ** 3 # [kg/m^3]
    cat_site_density = 2 * 1.41e-9 * units.mole / units.centimeter ** 2 # [kmol/m^2]
    area_spec = 42 * units.meter ** 2 / units.gram # [m^2/kg]
    # Calculate catalyst dispersion.
    cat_dispersion = area_spec * cat_site_density * cat_mol_weight
    # Set the volumetric flow rate from gas hour space velocity.
    gas_hour_space_vel = 24000 * 1 / units.hour # [1/s]
    vol_flow_rate = gas_hour_space_vel * (cat_mass / cat_density) # [m^3/s]
    # Set the cross section and the total volume of the reactor.
    reactor_length = 1.00 * units.centimeter # [m]
    diameter = 1.00 * units.millimeter # [m]
    # Calculate reactor volume and gas velocity.
    cross_section = np.pi * (diameter ** 2) / 4 # [m^2]
    reactor_volume = cross_section * reactor_length # [m^3]
    gas_velocity = vol_flow_rate / cross_section # [m/s]
    # Calculate the catalyst active area per unit of reactor volume.
    cat_moles = cat_mass / cat_mol_weight # [kmol]
    cat_area = cat_moles * cat_dispersion / cat_site_density # [m^2]
    alpha_cat = cat_area / reactor_volume # [m^2/m^3]
    
    # Get adsorbates and transition state atoms.
    with connect(name=db_ads_name) as db_ase:
        atoms_ads_list = get_atoms_list_from_db(db_ase=db_ase, surface=surface)
    with connect(name=db_ts_name) as db_ase:
        atoms_ts_list = get_atoms_list_from_db(db_ase=db_ase, surface=surface)
    # Get formations energies.
    names_ads_list, names_ts_list = get_names_list_from_yaml(yaml_file=yaml_file)
    e_form_dict = get_e_form_from_ase_atoms(
        atoms_ads_list=atoms_ads_list,
        atoms_ts_list=atoms_ts_list,
        names_ads_list=names_ads_list,
        names_ts_list=names_ts_list,
        e_form_key="E_form" if e_index is None else "E_form_list",
        e_index=e_index,
        species_key="species",
        free_sites=["(O)", "(Zr)", "(Ga)"],
        e_form_missing=np.nan,
        verbose=False,
    )
    # Missing formation energies.
    e_form_dict["CO2(Zr,Zr,Zr,O)"] = -1.674 # [eV]
    e_form_dict["H(Zr)+H(O)"] = -0.074 # [eV]
    e_form_dict["H2 + (Zr) + (O) <=> H(Zr)+H(O)"] = 0.267 # [eV]

    # Check for missing formation energies.
    if any(np.isnan(list(e_form_dict.values()))):
        raise ValueError("Missing formation energies in e_form_dict.")
    
    # Reaction mechanism.
    gas, cat, cat_ts = get_mechanism_from_yaml(
        yaml_file=yaml_file,
        e_form_dict=e_form_dict,
        site_density=cat_site_density,
        temperature=temperature,
        pressure=pressure,
        units_energy=units.eV / units.molecule,
    )
    gas.TPX = temperature, pressure, gas_molfracs_inlet
    cat.coverages = cat_coverages_inlet

    # Calculate gas space velocity.
    mol_flow_rate = vol_flow_rate * gas.density / gas.mean_molecular_weight # [kmol/s]
    cat_moles = cat_site_density * alpha_cat * reactor_volume # [kmol]
    gas_space_velocity = mol_flow_rate / cat_moles # [1/s]
    # Create an ideal reactor.
    cstr_length = reactor_length / n_cstr # [m]
    cstr_volume = cross_section * cstr_length # [m^3]
    cstr_cat_area = alpha_cat * cstr_volume # [m^2]
    mass_flow_rate = vol_flow_rate * gas.density # [kg/s]
    cstr = ct.IdealGasReactor(gas, energy="off", name="cstr")
    cstr.volume = cstr_volume
    surf = ct.ReactorSurface(cat, cstr, A=cstr_cat_area)
    # Add mass flow and pressure controllers.
    upstream = ct.Reservoir(gas, name="upstream")
    master = ct.MassFlowController(
        upstream=upstream,
        downstream=cstr,
        mdot=mass_flow_rate,
    )
    downstream = ct.Reservoir(gas, name="downstream")
    pcontrol = ct.PressureController(
        upstream=cstr,
        downstream=downstream,
        master=master,
        K=1e-6,
    )
    # Define the parameters of the simulation.
    sim = ct.ReactorNet([cstr])
    sim.rtol = 1.0e-12
    sim.atol = 1.0e-18
    sim.max_steps = 1e9
    sim.max_err_test_fails = 1e9
    sim.max_time_step = 1.0

    # Integrate the PFR along the reactor length (z).
    print("\n- Reactor integration.")
    gas_array = ct.SolutionArray(gas, extra=["z_reactor"])
    cat_array = ct.SolutionArray(cat, extra=["z_reactor"])
    # Print the header of the table.
    if print_molfracs is True:
        string = "distance[m]".rjust(14)
        for spec in gas.species_names:
            string += ("x_" + spec + "[-]").rjust(12)
        print(string)
    # Integrate the reactor.
    for ii in range(n_cstr + 1):
        z_reactor = ii * cstr_length
        gas_array.append(state=gas.state, z_reactor=z_reactor)
        cat_array.append(state=cat.state, z_reactor=z_reactor)
        # Print the state of the reactor.
        if print_molfracs is True:
            string = f"  {z_reactor:12f}"
            for spec in gas.species_names:
                string += f"  {gas[spec].X[0]:10f}"
            print(string)
        # Intergate the microkinetic model.
        if ii < n_cstr:
            advance_sim_to_steady_state(sim=sim, n_try_max=1000)
        # Set next CSTR inlet gas composition equal to the outlet of the previous CSTR.
        gas.TDY = cstr.thermo.TDY
        upstream.syncState()

    # Calculate CO2 conversion and the CH3OH productivity.
    print("\n- Catalytic activity.")
    gas_prod = "CH3OH"
    massfracs_in = get_Y_dict(gas_array[0])
    massfracs_out = get_Y_dict(gas_array[-1])
    conversion_CO2 = (massfracs_in["CO2"] - massfracs_out["CO2"]) / massfracs_in["CO2"]
    productiv_CH3OH = massfracs_out["CH3OH"] * mass_flow_rate / cat_mass
    productiv_CH3OH /= units.milligram / units.gram / units.hour
    print(f"Conversion of CO2     = {conversion_CO2 * 100:+12.6f} [%]")
    print(f"Productivity of CH3OH = {productiv_CH3OH:+12.6f} [mg/g_cat/hour]")
    
    # Get the coverages.
    coverages = {spec: cat[spec].coverages[0] for spec in cat.species_names}
    if print_coverages is True:
        print("\n- Coverages.")
        for spec, coverage in coverages.items():
            print(f"{spec:<30} {coverage:7.4e}")
    
    # Calculate elements molar balance.
    if element_balance is True:
        delta_moles_fracts = molar_balance_of_element(
            gas_in=gas_array[0],
            gas_out=gas_array[-1],
            mass=mass_flow_rate,
            return_fraction=True,
        )
        print("\n- Molar balances of elements.")
        for elem, delta_molfract in delta_moles_fracts.items():
            print(f"Balance of {elem} = {delta_molfract * 100:+7.2f} [%]")
    
    # Calculate the degree of rate control.
    if calculate_DRC is True:
        # Reactions.
        DRC_reactions = degree_rate_control(
            gas=gas,
            cat=cat,
            sim=sim,
            mdot=mass_flow_rate,
            upstream=upstream,
            gas_spec_target=gas_prod,
            multip_value=1.05,
            return_dict=True,
        )
        # Print DRC results.
        print("\n- Degree of rate control (reactions).")
        for ii, react in enumerate(DRC_reactions):
            print(f" {ii:3d} {react:80s} {DRC_reactions[react]:+7.4f}")
        sum_DRC = np.sum([DRC_reactions[react] for react in DRC_reactions])
        print(f"Sum DRC = {sum_DRC:+7.4f}")
        # Adsorbates.
        DRC_adsorbates = generalized_degree_rate_control(
            gas=gas,
            cat=cat,
            cat_ts=cat_ts,
            sim=sim,
            mdot=mass_flow_rate,
            upstream=upstream,
            gas_spec_target=gas_prod,
            delta_e=0.001,
            reactions_DRC=False,
            return_dict=True,
        )
        # Print DRC results.
        print("\n- Degree of rate control (adsorbates).")
        for ii, react in enumerate(DRC_adsorbates):
            print(f" {ii:3d} {react:80s} {DRC_adsorbates[react]:+7.4f}")
        # Merge DRC results.
        DRC_dict = {**DRC_reactions, **DRC_adsorbates}

    print("\n- Reaction path analysis.")
    # Read RPA keys from YAML file.
    with open("yaml/rpa_keys.yaml", "r") as fileobj:
        rpa_keys = yaml.safe_load(fileobj)
    # Run RPA.
    rpa_dict = reaction_path_analysis(
        gas=gas,
        cat=cat,
        surf=surf,
        filename=None,
    )
    # Print RPA results.
    for spec in rpa_keys:
        print(f"- {spec:40s} R[%]")
        for path, name in rpa_keys[spec].items():
            prod_perc = rpa_dict[spec][name]["prod_perc"]
            print(f"{path:40s} {prod_perc:+7.2f}")

    # Return results.
    return {
        "conversion_CO2": conversion_CO2,
        "productiv_CH3OH": productiv_CH3OH,
        "coverages": coverages,
        "DRC_dict": DRC_dict if calculate_DRC is True else None,
    }

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
