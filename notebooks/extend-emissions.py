# ---
# jupyter:
#   jupytext:
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
# # Extend emissions to 2025
#
# We don't yet have emissions beyond 2023, so make some stylised extensions.
#
# This is using fair calibration v1.4.5 and fair v2.2.4

# %%
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import CubicSpline

# %%
df_emis = pd.read_csv('../data/emissions/historical_1750-2023.csv')

# %%
df_emis

# %%
# we can crib this to make our actual emissions dataset
df_emis_ext = df_emis.copy()

# %%
# make new columns
df_emis_ext['2024.5'] = pd.Series(np.ones(51) * np.nan, index=df_emis.index)
df_emis_ext['2025.5'] = pd.Series(np.ones(51) * np.nan, index=df_emis.index)

# %%
df_emis_ext

# %%
# let's see what we still need to extrapolate
# Find all the species that don't have a 2024 or 2025 value; these need extrapolating
last_valid_year = {}
for variable in df_emis.variable:
    lvy = df_emis.loc[df_emis['variable']==variable, '1750.5':].squeeze(axis=0).last_valid_index()
    if lvy != '2025.5':  # ignore complete variables
        last_valid_year[variable] = lvy

# %%
last_valid_year

# %%
fig, ax = pl.subplots(8, 7, figsize = (16, 25))
ivar = 0
for variable in last_valid_year:
    ax[ivar//7, ivar%7].plot(
        np.arange(2000.5, float(last_valid_year[variable]) + 1), 
        df_emis.loc[df_emis['variable']==variable, '2000.5':last_valid_year[variable]].T,
        color='sienna',
        label='ScenarioMIP'
    )
    trend = (
        df_emis.loc[df_emis['variable']==variable, last_valid_year[variable]] -
        df_emis.loc[df_emis['variable']==variable, str(float(last_valid_year[variable])-1)]
    ).values[0]
    ax[ivar//7, ivar%7].plot(
        np.arange(float(last_valid_year[variable]), 2026.5, 1), 
        (
            df_emis.loc[df_emis['variable']==variable, last_valid_year[variable]].values[0] + 
            trend * (np.arange(2026.5-float(last_valid_year[variable])))
        ),
        color='skyblue',
        label='cubic spline'
    )
    ax[ivar//7, ivar%7].set_title(variable)
    ivar = ivar + 1
fig.tight_layout()

# %%
# let's implement
fig, ax = pl.subplots(8, 7, figsize = (16, 25))
ivar = 0
for variable in last_valid_year:
    trend = (
        df_emis.loc[df_emis['variable']==variable, last_valid_year[variable]] -
        df_emis.loc[df_emis['variable']==variable, str(float(last_valid_year[variable])-1)]
    ).values[0]
    df_emis_ext.loc[df_emis['variable']==variable, str(float(last_valid_year[variable])):'2025.5'] = (
        df_emis_ext.loc[df_emis['variable']==variable, last_valid_year[variable]].values[0] + 
        trend * (np.arange(2026.5-float(last_valid_year[variable])))
    )
    ax[ivar//7, ivar%7].plot(
        np.arange(2000.5, 2026.5), 
        df_emis_ext.loc[df_emis_ext['variable']==variable, '2000.5':'2025.5'].T,
        color='skyblue',
    )
    ax[ivar//7, ivar%7].set_title(variable)
    ivar = ivar + 1
fig.tight_layout()

# %%
df_emis_ext

# %%
df_emis_ext.to_csv('../output/emissions_extrapolated_1750-2025.csv', index=False)

# %%
