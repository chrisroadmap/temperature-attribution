# temperature-attribution
Attribution of contributions to global mean surface temperature change through to 2023.

Uses `fair-calibrate` v1.4.1, where emissions and forcing are historical best estimates. I use a one-year extrapolation of emissions to 2023 following the draft ScenarioMIP "M" pathway (the choice really should not matter).

# requirements
- `python`
- `conda` or `mamba`

Obtain a minimal distribution of `conda` from https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html. `conda` also ships with `python`, so you do not need to install `python` separately.

# reproduction
1. clone the repository and `cd` to `temperature-attribution`
2. create the environment using `conda env create -f environment.yml`
3. activate the environment with `conda activate temperature-attribution`
4. run `nbstripout --install` to ensure clean notebook commits
5. fire up `jupyter notebook` and run the notebook in the `notebooks` directory.

Plots and outputs are in... not surprisingly, the `plots` and `output` directories respectively.
