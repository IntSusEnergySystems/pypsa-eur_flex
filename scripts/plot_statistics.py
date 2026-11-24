# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plots bar charts of key statistics per technology carrier from a solved network.

Reads the solved network and computes capacity factors, installed and optimal
capacities, capital and operational expenditure, curtailment, supply, withdrawal
and market values with the PyPSA statistics module. Each statistic is drawn as
a horizontal bar chart coloured by carrier, with loads shown as a carrier of
their own. The charts give a quick sanity check of a single scenario's results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns
from matplotlib.axes import Axes

from scripts._helpers import configure_logging, set_scenario_config

sns.set_theme("paper", style="whitegrid")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_base_statistics")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    n.loads.carrier = "load"
    if "load" not in n.carriers.index:
        n.add("Carrier", "load")
    n.carriers.loc["load", ["nice_name", "color"]] = "Load", "darkred"
    colors = n.carriers.color.astype("object").replace("", pd.NA).fillna("lightgrey")
    colors.update(
        {
            "csp": "#e07a00",
            "marine": "#167a8a",
            "rural coal boiler": "#4e4e4e",
            "urban decentral coal boiler": "#5a5a5a",
            "coal boiler": "#4e4e4e",
            "other energy coal": "#3f3f3f",
            "other energy gas": "#c45100",
            "other energy oil": "#8a8a8a",
            "none": "#cccccc",
        }
    )

    def rename_index(ds: pd.Series) -> pd.Series:
        specific = ds.index.map(lambda x: f"{x[1]}\n({x[0]})")
        generic = ds.index.get_level_values("carrier")
        duplicated = generic.duplicated(keep=False)
        index = specific.where(duplicated, generic)
        return ds.set_axis(index)

    def plot_static_per_carrier(
        ds: pd.Series, ax: Axes, drop_zero: bool = True
    ) -> None:
        if drop_zero:
            ds = ds[ds != 0]
        ds = ds.dropna()
        c = colors.reindex(ds.index.get_level_values("carrier")).fillna("lightgrey")
        ds = ds.pipe(rename_index)
        label = f"{ds.attrs['name']} [{ds.attrs['unit']}]"
        ds.plot.barh(color=c.values, xlabel=label, ax=ax)
        ax.grid(axis="y")

    fig, ax = plt.subplots()
    ds = n.statistics.capacity_factor().dropna()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.capacity_factor_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.installed_capacity().dropna()
    ds = ds.drop("Line")
    ds = ds.drop(("Generator", "Load"), errors="ignore")
    ds = ds / 1e3
    ds.attrs["unit"] = "GW"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.installed_capacity_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.optimal_capacity()
    ds = ds.drop("Line")
    ds = ds.drop(("Generator", "Load"), errors="ignore")
    ds = ds / 1e3
    ds.attrs["unit"] = "GW"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.optimal_capacity_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.capex()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.capital_expenditure_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.opex()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.operational_expenditure_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.curtailment()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.curtailment_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.supply()
    ds = ds.drop("Line")
    ds = ds / 1e6
    ds.attrs["unit"] = "TWh"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.supply_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.withdrawal()
    ds = ds.drop("Line")
    ds = ds / -1e6
    ds.attrs["unit"] = "TWh"
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.withdrawal_bar)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds = n.statistics.market_value()
    plot_static_per_carrier(ds, ax)
    fig.savefig(snakemake.output.market_value_bar)
    plt.close(fig)

    # touch file
    with open(snakemake.output.barplots_touch, "a"):
        pass
