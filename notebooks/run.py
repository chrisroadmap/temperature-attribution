# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run temperature attributions and save out
#
# 1. Run one scenario emissions driven. Save out the following forcing categories:
#
# - All forcing
# - All minus GHGs
# - All minus aerosols
# - All minus other anthropogenic
# - All minus natural
# - All minus anthropogenic
#
# 2. re-run forcing-driven scenarios with the above.

# %%
import os

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

# %%
pl.style.use('../defaults.mplstyle')

# %%
output_ensemble_size=841

# %%
scenarios = ["all"]

# %%
f = FAIR()
f.define_time(1750, 2026, 1)
f.define_scenarios(scenarios)
species, properties = read_properties('../data/calibration/v1.4.5/species_configs_properties_1.4.5.csv')
f.define_species(species, properties)
f.ch4_method='Thornhill2021'
df_configs = pd.read_csv('../data/calibration/v1.4.5/calibrated_constrained_parameters_1.4.5.csv', index_col=0)
f.define_configs(df_configs.index)
f.allocate()

# %% [markdown]
# ## First, emissions-driven run

# %% [markdown]
# ### Get emissions and forcing into fair
#
# We use the 2022 harmonization of historical emissions from Smith et al. (2024), plus a three-year extension to 2025.

# %%
f.fill_from_csv(
    emissions_file='../output/emissions_extrapolated_1750-2025.csv',
    forcing_file='../data/forcing/volcanic_solar.csv',
)

# %%
f.emissions

# %% [markdown]
# ### Fill in all the configs

# %%
fill(
    f.forcing,
    f.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    f.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

f.fill_species_configs("../data/calibration/v1.4.5/species_configs_properties_1.4.5.csv")
f.override_defaults("../data/calibration/v1.4.5/calibrated_constrained_parameters_1.4.5.csv")

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
f.run()

# %% [markdown]
# ## Now isolate out forcing cats

# %%
f.species

# %%
ghgs = [
 'CO2',
 'CH4',
 'N2O',
 'CFC-11',
 'CFC-12',
 'CFC-113',
 'CFC-114',
 'CFC-115',
 'HCFC-22',
 'HCFC-141b',
 'HCFC-142b',
 'CCl4',
 'CHCl3',
 'CH2Cl2',
 'CH3Cl',
 'CH3CCl3',
 'CH3Br',
 'Halon-1211',
 'Halon-1301',
 'Halon-2402',
 'CF4',
 'C2F6',
 'C3F8',
 'c-C4F8',
 'C4F10',
 'C5F12',
 'C6F14',
 'C7F16',
 'C8F18',
 'NF3',
 'SF6',
 'SO2F2',
 'HFC-125',
 'HFC-134a',
 'HFC-143a',
 'HFC-152a',
 'HFC-227ea',
 'HFC-23',
 'HFC-236fa',
 'HFC-245fa',
 'HFC-32',
 'HFC-365mfc',
 'HFC-4310mee',
]

# %%
aerosols = [
 'Aerosol-radiation interactions',
 'Aerosol-cloud interactions',
]

# %%
natural = [
 'Solar',
 'Volcanic',
]

# %%
other = [
 'Ozone',
 'Light absorbing particles on snow and ice',
 'Stratospheric water vapour',
 'Land use',
]

# %%
anthro = list(set(f.species) - set(natural))
anthro

# %%
pl.plot(f.forcing.sel(specie=ghgs).sum(dim='specie').sel(scenario='all'));

# %%
pl.plot(f.forcing.sel(specie=aerosols).sum(dim='specie').sel(scenario='all'));

# %%
pl.plot(f.forcing.sel(specie=other).sum(dim='specie').sel(scenario='all'));

# %%
pl.plot(f.forcing.sel(specie=natural).sum(dim='specie').sel(scenario='all'));

# %%
pl.plot(f.forcing.sel(specie=anthro).sum(dim='specie').sel(scenario='all'));

# %% [markdown]
# ### Check linearity
#
# note y-axis value very small - minimal floating point differences are fine

# %%
pl.plot(
    f.forcing_sum.sel(scenario='all') - (
        f.forcing.sel(specie=ghgs).sum(dim='specie').sel(scenario='all') +
        f.forcing.sel(specie=aerosols).sum(dim='specie').sel(scenario='all') +
        f.forcing.sel(specie=other).sum(dim='specie').sel(scenario='all') + 
        f.forcing.sel(specie=natural).sum(dim='specie').sel(scenario='all')
    )
);

# %%
pl.plot(
    f.forcing_sum.sel(scenario='all') - (
        f.forcing.sel(specie=anthro).sum(dim='specie').sel(scenario='all') +
        f.forcing.sel(specie=natural).sum(dim='specie').sel(scenario='all')
    )
);


# %% [markdown]
# ## Constrained ensemble statistics
#
# Compare the constrained ensemble to the IPCC assessed ranges. ECS is asymmetric so we fit a skew-normal to the percentiles.

# %%
def opt(x, q05_desired, q50_desired, q95_desired):
    """Fit a skew-normal distribution to 5, 50, 95 percentiles.
    
    x is (a, loc, scale) in that order."""
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


# %%
# conversion from one standard deviation to 90% range
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
NINETY_TO_ONESIGMA

# %%
ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x
ecs_params

# %%
aer_dist = scipy.stats.norm(loc=-1.3, scale=0.7/NINETY_TO_ONESIGMA)

# %%
ecs_samp = ((df_configs['forcing_4co2']/2)/df_configs['ocean_heat_transfer[0]']).values
ecs_dist = scipy.stats.skewnorm(a=ecs_params[0], loc=ecs_params[1], scale=ecs_params[2])

# %%
aer_samp = f.forcing.sel(specie=aerosols, timebounds=np.arange(2005, 2015)).sum(dim='specie').sel(scenario='all').mean(dim='timebounds')

# %%
fig, ax = pl.subplots(1, 3, figsize=(12,4))

ax[0].hist(aer_samp, bins=np.arange(-2.5, 0.1, 0.1), density=True)
ax[0].plot(np.linspace(-3, 0.5, 101), aer_dist.pdf(np.linspace(-3, 0.5, 101)), color='k', lw=2)
ax[0].set_xlim(-3, 0.5)
ax[0].set_title('Aerosol forcing')
ax[0].set_xlabel('W m$^{-2}$')
ax[0].set_ylabel('Density')

ax[1].hist(ecs_samp, bins=np.arange(0, 8.1, 0.2), density=True, label='Constrained ensemble')
ax[1].plot(np.linspace(0, 8, 101), ecs_dist.pdf(np.linspace(0, 8, 101)), color='k', lw=2, label='IPCC AR6')
ax[1].set_xlim(0, 8)
ax[1].legend(fontsize=10, frameon=False)
ax[1].set_title('ECS')
ax[1].set_xlabel('ECS, °C')
ax[1].set_ylabel('Density')

ax[2].scatter(ecs_samp, aer_samp)
ax[2].set_xlim(1,8)
ax[2].set_ylim(-2.6, 0.1)
ax[2].set_title('Correlation')
ax[2].set_xlabel('ECS, °C')
ax[2].set_ylabel('Aerosol forcing, W m$^{-2}$')

fig.tight_layout()
os.makedirs('../plots/', exist_ok=True)
pl.savefig('../plots/distributions.png')

# %%
scenarios = ["all", "no_ghgs", "no_aerosols", "no_other", "no_natural", "no_anthro"]

# %%
ff = FAIR()
ff.define_time(1750, 2026, 1)
ff.define_scenarios(scenarios)
ff.define_configs(df_configs.index)

species = ["bulk"]
properties = {
    "bulk": {
        "type": "unspecified",
        "input_mode": "forcing",
        "greenhouse_gas": False,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    }
}

ff.define_species(species, properties)
ff.allocate()

# %% [markdown]
# ### Create forcing-driven time series

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=ghgs).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_ghgs"
)

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=aerosols).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_aerosols"
)

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=other).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_other"
)

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=natural).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_natural"
)

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=anthro).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_anthro"
)

# %%
fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all'),
    specie="bulk",
    scenario="all"
)

# %% [markdown]
# ### fill and initialise configs

# %%
# climate response
ff.override_defaults("../data/calibration/v1.4.5/calibrated_constrained_parameters_1.4.5.csv")

# initial conditions
initialise(ff.forcing, 0)
initialise(ff.temperature, 0)
initialise(ff.ocean_heat_content_change, 0)

# %%
ff.run()

# %%
base = np.arange(1850, 1901)
temp_ghgs = (
    (
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_ghgs", layer=0) - ff.temperature.sel(scenario="no_ghgs", layer=0, timebounds=base).mean(dim='timebounds')
    )
)
temp_aerosols = (
    (
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_aerosols", layer=0) - ff.temperature.sel(scenario="no_aerosols", layer=0, timebounds=base).mean(dim='timebounds')
    )
)
temp_natural = (
    (
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_natural", layer=0) - ff.temperature.sel(scenario="no_natural", layer=0, timebounds=base).mean(dim='timebounds')
    )
)
temp_other = (
    (
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_other", layer=0) - ff.temperature.sel(scenario="no_other", layer=0, timebounds=base).mean(dim='timebounds')
    )
)
temp_anthro = (
    (
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_anthro", layer=0) - ff.temperature.sel(scenario="no_anthro", layer=0, timebounds=base).mean(dim='timebounds')
    )
)
temp_all = ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')

# %%
df_obs = pd.read_csv('../data/observations/HadCRUT.5.1.0.0.analysis.summary_series.global.annual.1850-2025_rebased_1850-1900.csv', index_col=0)
df_obs

# %% [markdown]
# ## Gillett et al. (2021) style plot
#
# fig 1 b in https://www.nature.com/articles/s41558-020-00965-9
#
# Note the ranges coming out of fair are much more constrained than CMIP6 models in Gillett et al., since the historical climate record and emergent climate metrics are constrained on observations and AR6 assessment (by the fair-calibrate ensemble).

# %%
pl.fill_between(np.arange(1750, 2027), temp_all.quantile(0.05, dim="config"), temp_all.quantile(0.95, dim="config"), color='orange', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_all.median(dim="config"), color='orange', label='All forcers');

pl.fill_between(np.arange(1750, 2027), temp_ghgs.quantile(0.05, dim="config"), temp_ghgs.quantile(0.95, dim="config"), color='0.5', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_ghgs.median(dim="config"), color='0.5', label='Greenhouse gases');

pl.fill_between(np.arange(1750, 2027), temp_aerosols.quantile(0.05, dim="config"), temp_aerosols.quantile(0.95, dim="config"), color='blue', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_aerosols.median(dim="config"), color='blue', label='Aerosols');

#pl.plot(np.arange(1750, 2024), temp_other.median(dim="config"), color='green')

pl.fill_between(np.arange(1750, 2027), temp_natural.quantile(0.05, dim="config"), temp_natural.quantile(0.95, dim="config"), color='green', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_natural.median(dim="config"), color='green', label='Natural');
pl.plot(df_obs.gmst, color='k', label='HadCRUT5')

pl.ylabel('°C relative to 1850-1900')
pl.legend();
pl.grid()

pl.xlim(1850, 2026)
pl.ylim(-1.5, 2.5)

os.makedirs('../plots/', exist_ok=True)
pl.tight_layout()
pl.savefig('../plots/attributed_warming_rel1850-1900.png')

# %%
pl.fill_between(np.arange(1750, 2027), temp_anthro.quantile(0.05, dim="config"), temp_anthro.quantile(0.95, dim="config"), color='purple', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_anthro.median(dim="config"), color='purple', label='Anthropogenic');

pl.fill_between(np.arange(1750, 2027), temp_natural.quantile(0.05, dim="config"), temp_natural.quantile(0.95, dim="config"), color='green', alpha=0.2, lw=1)
pl.plot(np.arange(1750, 2027), temp_natural.median(dim="config"), color='green', label='Natural');
pl.plot(df_obs.gmst, color='k', label='HadCRUT5')

pl.ylabel('°C relative to 1850-1900')
pl.legend();
pl.grid()

pl.xlim(1850, 2026)
pl.ylim(-0.4, 1.6)

os.makedirs('../plots/', exist_ok=True)
pl.tight_layout()
pl.savefig('../plots/attributed_warming_anth_nat_rel1850-1900.png')

# %%
temp_aerosols.median(dim="config")

# %%
df_out = pd.DataFrame(
    np.array(
        [
            temp_ghgs.quantile(0.05, dim="config").data,
            temp_ghgs.median(dim="config").data,
            temp_ghgs.quantile(0.95, dim="config").data,
            temp_aerosols.quantile(0.05, dim="config").data,
            temp_aerosols.median(dim="config").data,
            temp_aerosols.quantile(0.95, dim="config").data,
            temp_other.quantile(0.05, dim="config").data,
            temp_other.median(dim="config").data,
            temp_other.quantile(0.95, dim="config").data,
            temp_natural.quantile(0.05, dim="config").data,
            temp_natural.median(dim="config").data,
            temp_natural.quantile(0.95, dim="config").data,
            temp_anthro.quantile(0.05, dim="config").data,
            temp_anthro.median(dim="config").data,
            temp_anthro.quantile(0.95, dim="config").data,
            temp_all.quantile(0.05, dim="config").data,
            temp_all.median(dim="config").data,
            temp_all.quantile(0.95, dim="config").data,
        ]
    ).T,
    index=np.arange(1750, 2027),
    columns=['ghg_05', 'ghg_50', 'ghg_95', 'aerosol_05', 'aerosol_50', 'aerosol_95', 'other_05', 'other_50', 'other_95', 'natural_05', 'natural_50', 'natural_95', 'anthro_05', 'anthro_50', 'anthro_95', 'all_05', 'all_50', 'all_95']
)

# %%
os.makedirs('../output', exist_ok=True)
df_out.to_csv('../output/attributed_warming.csv')

# %%
