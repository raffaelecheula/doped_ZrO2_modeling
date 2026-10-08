# Graph-based modeling of CO₂ hydrogenation on doped ZrO₂

This repository contains the Python scripts, atomistic databases, and configuration files used to reproduce the computational workflow of the study **“Improved screening of doped ZrO₂ catalysts for CO₂ hydrogenation from graph-based machine learning”**.

The workflow combines density functional theory (DFT) data, machine-learning predictions of adsorption and reaction energies, and microkinetic modeling to investigate CO₂ hydrogenation to methanol on doped ZrO₂ surfaces.

## Overview

The project evaluates graph-based machine-learning models for predicting the energetics of surface intermediates and transition states across doped ZrO₂ catalysts. Model performance is assessed using cross-validation, including tests that group data by surface, and compared with conventional linear scaling approaches such as transition-state scaling relations (TSR) and Brønsted–Evans–Polanyi (BEP) relations.

The predicted energetics are then incorporated into microkinetic models to evaluate methanol production under different temperatures and pressures, and to compare model-based predictions against DFT-based results.

The workflow covers:

- Preparation of DFT-derived ASE databases and atomic features.
- Construction of the reaction mechanism and thermochemical data.
- Cross-validation of linear and graph-based energy-prediction models.
- Integration of predicted energies into microkinetic simulations.
- Analysis of methanol yields, prediction errors, degrees of rate control, and reaction pathways.

### Scripts

| Script | Purpose |
| --- | --- |
| `1_prepare_database.py` | Prepare the adsorbate and transition-state databases from DFT calculations, including atomic descriptors and features. |
| `2_prepare_mechanism.py` | Prepare the reaction mechanism and thermochemical parameters used for microkinetic simulations. |
| `3_crossvalidation.py` | Train and validate linear and graph-based models, including stratified and group-based cross-validation. |
| `4_microkinetics_integration.py` | Incorporate DFT or model-predicted energies into microkinetic simulations and calculate catalytic performance. |
| `5_plot_activity_comparison.py` | Compare model-predicted and DFT-based methanol yields. |
| `6_plot_errors_and_drcs.py` | Analyze prediction errors and degrees of rate control. |
| `7_plot_reaction_paths.py` | Visualize the contributions of different reaction pathways. |

The `databases/` directory contains ASE databases, including reference DFT data and model-generated results. The `yaml/` directory contains surface definitions, species and reaction information, thermochemical parameters, and plotting settings.

## Requirements

The scripts are written in Python and use the following principal libraries:

- [ASE](https://wiki.fysik.dtu.dk/ase/) — atomic structures and databases.
- [NumPy](https://numpy.org/) — numerical calculations.
- [Matplotlib](https://matplotlib.org/) — visualization.
- [PyYAML](https://pyyaml.org/) — configuration files and results.
- [Cantera](https://cantera.org/) — chemical kinetics simulations.
- [ase_ml_models](https://github.com/raffaelecheula/ase_cantera_microkinetics) — feature generation and machine-learning workflows.
- [ase_cantera_microkinetics](https://github.com/raffaelecheula/ase_cantera_microkinetics) — preparation and integration of microkinetic models.

The last two packages are required by the scripts but are not included in this repository. Install them in your Python environment before running the full workflow.

## Usage

Clone the repository and navigate to the `scripts/` directory:

```bash
git clone https://github.com/raffaelecheula/doped_ZrO2_modeling.git
cd doped_ZrO2_modeling/scripts
```

The scripts use paths relative to the `scripts/` directory and are intended to be run from there. A typical workflow is:

```bash
python 1_prepare_database.py
python 2_prepare_mechanism.py
python 3_crossvalidation.py
python 4_microkinetics_integration.py
python 5_plot_activity_comparison.py
python 6_plot_errors_and_drcs.py
python 7_plot_reaction_paths.py
```

## Citation

If you use this repository in your research, please cite the associated study:

> Jensen, K. W., Andersen, M., and Cheula, R. *Improved screening of doped ZrO₂ catalysts for CO₂ hydrogenation from graph-based machine learning*.

Publication details and DOI should be added here when available.

## License

This project is distributed under the [GNU General Public License v3.0](LICENSE).
