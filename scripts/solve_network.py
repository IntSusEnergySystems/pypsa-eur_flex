# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
[documentation of PyPSA](https://docs.pypsa.org/latest/user-guide/network-optimization/).

The optimization is based on the `network.optimize` function.
Additionally, some extra constraints specified in [solve_network][] are added.
"""

import importlib
import logging
import os
import re
import sys
from functools import partial
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from linopy.constants import SolverStatus, TerminationCondition
from pypsa.descriptors import get_activity_mask

from scripts._benchmark import memory_logger
from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    get,
    set_scenario_config,
)
from scripts.co2_budget import national_co2_fraction

logger = logging.getLogger(__name__)

# Allow for PyPSA versions <0.35
if PYPSA_V1:
    pypsa.network.power_flow.logger.setLevel(logging.WARNING)
else:
    pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass


# TODO: move to compose_network
def add_land_use_constraint(n: pypsa.Network) -> None:
    """
    Add global constraints for tech capacity limit.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance

    Returns
    -------
    pypsa.Network
        Network with added land use constraints
    """
    logger.info("Add land-use constraint for perfect foresight")

    def check_p_min_p_max(p_nom_max: pd.Series) -> None:
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: (
            f"nom_max_{row['carrier']}"
            + (f"_{row['build_year']}" if row["build_year"] is not None else "")
        ),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values


def add_solar_potential_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.

    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {} if PYPSA_V1 else {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def add_co2_sequestration_limit(
    n: pypsa.Network,
    limit_dict: dict[str, float],
    planning_horizons: str | None,
) -> None:
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    limit_dict : dict[str, float]
        CO2 sequestration potential limit constraints by year.
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """

    if not n.investment_periods.empty:
        nyears = n.snapshot_weightings.groupby(level="period").generators.sum() / 8760
        periods = n.investment_periods
        limit = pd.Series(
            {period: nyears[period] * get(limit_dict, period) for period in periods}
        )
        limit.index = limit.index.map(lambda s: f"co2_sequestration_limit-{s}")
        names = limit.index
    else:
        nyears = n.snapshot_weightings.generators.sum() / 8760
        limit = get(limit_dict, int(planning_horizons)) * nyears
        periods = np.nan
        names = "co2_sequestration_limit"

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


VALID_GLC_SENSES = {"<=", ">=", "=="}


def get_glc_sense(glc: pd.Series) -> str:
    """Return the validated sense of a global constraint."""
    sense = glc.loc["sense"]
    if sense not in VALID_GLC_SENSES:
        raise ValueError(
            f"Unsupported sense {sense!r} for global constraint {glc.name!r}. "
            f"Expected one of {sorted(VALID_GLC_SENSES)}."
        )
    return sense


def emission_stores(n: pypsa.Network, emissions: pd.Index) -> pd.DataFrame:
    """Return the non-cyclic stores that accumulate emissions on their bus."""
    bus_carrier = n.stores.bus.map(n.buses.carrier)
    stores = n.stores[bus_carrier.isin(emissions) & ~n.stores.e_cyclic]
    seeded = stores.index[stores.e_initial.ne(0) | stores.e_initial_per_period]
    if not seeded.empty:
        raise ValueError(
            f"Emission stores {list(seeded)} start from a non-zero level "
            "(e_initial or e_initial_per_period). The CO2 constraints read the "
            "store level as cumulative emissions, which only holds from zero."
        )
    return stores


def add_carbon_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def add_carbon_budget_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def add_max_growth(n: pypsa.Network, opts: dict) -> None:
    """
    Add maximum growth rates for different carriers.
    """

    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"]:
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"]:
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period


def add_retrofit_gas_boiler_constraint(
    n: pypsa.Network, snapshots: pd.DatetimeIndex
) -> None:
    """
    Allow retrofitting of existing gas boilers to H2 boilers and impose load-following must-run condition on existing gas boilers.
    Modifies the network in place, no return value.

    n : pypsa.Network
        The PyPSA network to be modified
    snapshots : pd.DatetimeIndex
        The snapshots of the network
    """
    c = "Link"
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    if gas_i.empty or h2_i.empty:
        return

    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    if PYPSA_V1:
        p_gas = dispatch.sel(name=gas_i)
        p_h2 = dispatch.sel(name=h2_i)
    else:
        p_gas = dispatch.sel(Link=gas_i)
        p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def enforce_autarky(n: pypsa.Network, only_crossborder: bool = False) -> None:
    """
    Remove transmission lines to enforce autarky.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    only_crossborder : bool
        If True, only remove cross-border connections. If False, remove all lines.
    """
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index

    logger.info(
        f"Enforcing autarky: removing {len(lines_rm)} lines and {len(links_rm)} links"
    )
    n.remove("Line", lines_rm)
    n.remove("Link", links_rm)


def add_load_balance_components(
    n: pypsa.Network, config: dict, sign: float = 1
) -> None:
    """
    Add load shedding or load sinks to the network with carrier 'load'.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to be modified.
    config : dict
        The load shedding or load sinks settings.
    sign : float
        Direction of the added generators. Positive for load shedding, negative for load sinks.

    Returns
    -------
    None
        Modifies PyPSA network in place.
    """
    if "load" not in n.carriers.index:
        n.add("Carrier", "load")

    carriers = config.get("carriers", {})
    default_cost = config.get("default_cost")
    balance_comp = "shedding" if sign > 0 else "sink"

    logger.info(
        f"Add load {balance_comp} for {'all carriers' if config.get('all_carriers') else ', '.join(carriers)}."
    )

    for bus_carrier, price in carriers.items():
        buses_i = n.buses[n.buses.carrier == bus_carrier].index
        n.add(
            "Generator",
            buses_i,
            f" load {balance_comp}",
            bus=buses_i,
            carrier="load",
            marginal_cost=price,
            p_nom=np.inf,
            sign=sign,
        )

    if config.get("all_carriers", False):
        buses_rest_i = n.buses[~n.buses.carrier.isin(carriers)].index
        n.add(
            "Generator",
            buses_rest_i,
            f" load {balance_comp}",
            bus=buses_rest_i,
            carrier="load",
            marginal_cost=default_cost,
            p_nom=np.inf,
            sign=sign,
        )


def prepare_network(
    n: pypsa.Network,
    config: dict,
    solve_opts: dict,
    foresight: str,
    planning_horizons: str | None,
    co2_sequestration_potential: dict[str, float],
    limit_max_growth: dict[str, Any] | None = None,
    rolling_horizon: bool = False,
) -> None:
    """
    Prepare network with various constraints and modifications.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary
    solve_opts : Dict
        Dictionary of solving options containing clip_p_max_pu, load_shedding etc.
    foresight : str
        Planning foresight type ('myopic' or 'perfect')
    planning_horizons : str or None
        The current planning horizon year or None for perfect foresight
    co2_sequestration_potential : Dict[str, float]
        CO2 sequestration potential constraints by year

    Returns
    -------
    pypsa.Network
        Modified PyPSA network with added constraints
    """
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df.abs() > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if (load_shedding := solve_opts.get("load_shedding", {})).get("enable", False):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        add_load_balance_components(n, load_shedding)

    if (load_sinks := solve_opts.get("load_sinks", {})).get("enable", False):
        add_load_balance_components(n, load_sinks, sign=-1)

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.components:
            # if 'capital_cost' in t.static:
            #    t.static['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.static)) - 0.5)
            if "marginal_cost" in t.static:
                t.static["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.static)) - 0.5
                )

        for t in n.components[["Line", "Link"]]:
            if t.static.empty:
                continue
            t.static["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.static)) - 0.5)
            ) * t.static["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "perfect":
        add_land_use_constraint(n)
        if limit_max_growth is not None and limit_max_growth["enable"]:
            add_max_growth(n, limit_max_growth)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(
            n, limit_dict=limit_dict, planning_horizons=planning_horizons
        )

    # Apply autarky constraint if configured
    autarky_cfg = config["electricity"]["autarky"]
    if autarky_cfg["enable"]:
        only_crossborder = autarky_cfg["by_country"]
        enforce_autarky(n, only_crossborder=only_crossborder)

    # rolling horizon disables cyclic storage
    if rolling_horizon:
        n.storage_units.cyclic_state_of_charge = False
        n.storage_units.state_of_charge_initial = 0
        n.stores.e_cyclic = False
        n.stores.e_initial = 0


def _ccl_horizon_block(table: pd.DataFrame, horizon: str | int) -> pd.DataFrame:
    """Return the min/max block for one planning year."""
    years = table.columns.get_level_values(0)
    present = set(years)
    key: str | int | None = None
    for candidate in (horizon, str(horizon)):
        if candidate in present:
            key = candidate
            break
    if key is None:
        try:
            as_int = int(horizon)
        except (TypeError, ValueError):
            as_int = None
        if as_int in present:
            key = as_int
    if key is None:
        raise KeyError(
            f"No capacity limits for {horizon}. Years in the file: "
            f"{sorted({str(year) for year in present})}."
        )
    block = table[key].copy()
    block.index = block.index.set_names(["country", "carrier"])
    return block


def _ccl_rename(table: pd.DataFrame, options: dict) -> dict[str, str]:
    """Map component carriers onto the aggregate names used in the limit file."""
    carriers = set(table.index.get_level_values("carrier"))
    rename: dict[str, str] = {}
    if options["agg_offwind"] or "offwind-all" in carriers:
        rename.update(
            {
                "offwind-ac": "offwind-all",
                "offwind-dc": "offwind-all",
                "offwind-float": "offwind-all",
                "offwind": "offwind-all",
            }
        )
    if options["agg_solar"] or "solar-all" in carriers:
        rename.update(
            {
                "solar": "solar-all",
                "solar-hsat": "solar-all",
                "solar rooftop": "solar-all",
            }
        )
    if "gas" in carriers:
        rename.update({"OCGT": "gas", "CCGT": "gas"})
    if "coal-all" in carriers:
        rename.update({"coal": "coal-all", "lignite": "coal-all"})
    return rename


def _ccl_existing_mw(
    n: pypsa.Network,
    component: str,
    rename: dict[str, str],
    year: int,
) -> pd.Series:
    """Electrical MW of non-extendable assets still online in ``year``."""
    static = n.generators if component == "Generator" else n.links
    fixed = static.loc[~static.p_nom_extendable].copy()
    if fixed.empty:
        return pd.Series(dtype=float, name="p_nom")
    if {"build_year", "lifetime"} <= set(fixed.columns):
        fixed = fixed.loc[(fixed["build_year"] + fixed["lifetime"]) >= year]
    if fixed.empty:
        return pd.Series(dtype=float, name="p_nom")
    bus = fixed.bus if component == "Generator" else fixed.bus1
    fixed["country"] = bus.map(n.buses.country)
    fixed["carrier"] = fixed.carrier.replace(rename)
    if component == "Link":
        fixed["p_nom"] = fixed.p_nom.fillna(0) * fixed.efficiency.fillna(1.0)
    grouped = (
        fixed.dropna(subset=["country"])
        .groupby(["country", "carrier"])["p_nom"]
        .sum()
    )
    return grouped.rename("p_nom")


def _ccl_extendable_expression(
    n: pypsa.Network,
    component: str,
    rename: dict[str, str],
):
    """Extendable capacity, in electrical MW, grouped by country and carrier."""
    variable = f"{component}-p_nom"
    if variable not in n.model.variables:
        return None
    static = n.generators if component == "Generator" else n.links
    extendable = static.query("p_nom_extendable").copy()
    if extendable.empty:
        return None
    if not PYPSA_V1:
        extendable = extendable.rename_axis(index=f"{component}-ext")
    extendable["carrier"] = extendable.carrier.replace(rename)
    bus = extendable.bus if component == "Generator" else extendable.bus1
    p_nom = n.model[variable]
    if component == "Link":
        p_nom = p_nom.loc[extendable.index] * extendable.efficiency.fillna(1.0)
    # The series name has to be ``bus`` so the grouped axis matches the
    # generator constraint and can be renamed to country.
    country = bus.map(n.buses.country)
    country.name = "bus"
    grouper = pd.concat([country, extendable.carrier], axis=1)
    return p_nom.groupby(grouper).sum().rename(bus="country")


def add_CCL_constraints(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add country and carrier capacity limits.

    Values in ``agg_p_nom_minmax.csv`` are electrical MW. Generator capacity is
    used as stored. Link capacity is the fuel-side ``p_nom`` times ``efficiency``,
    which is how coal, lignite, oil, OCGT and CCGT are represented once sector
    coupling is on. ``gas`` sums OCGT and CCGT. ``coal-all`` sums coal and lignite.
    Wind and solar aggregates follow ``agg_offwind`` and ``agg_solar``.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """

    assert planning_horizons is not None, (
        "add_CCL_constraints are not implemented for perfect foresight, yet"
    )

    options = config["solving"]["agg_p_nom_limits"]
    table = pd.read_csv(options["file"], index_col=[0, 1], header=[0, 1])
    limits = _ccl_horizon_block(table, planning_horizons)
    rename = _ccl_rename(limits, options)
    logger.info("Adding generation capacity constraints per carrier and country")
    year = int(planning_horizons)
    wrote = False

    for component in ("Generator", "Link"):
        lhs = _ccl_extendable_expression(n, component, rename)
        if lhs is None:
            continue
        carriers = set(lhs.indexes["group"].get_level_values("carrier"))
        component_limits = limits.loc[
            limits.index.get_level_values("carrier").isin(carriers)
        ]
        if component_limits.empty:
            continue
        if options["include_existing"]:
            existing = _ccl_existing_mw(n, component, rename, year)
            if existing.empty:
                existing = pd.DataFrame(
                    columns=["p_nom"],
                    index=pd.MultiIndex.from_tuples([], names=["country", "carrier"]),
                )
            else:
                existing = existing.to_frame()
                existing.index = existing.index.set_names(["country", "carrier"])

            def _remaining(bound: str) -> xr.DataArray:
                limit = component_limits[bound].dropna()
                idx = limit.index.join(existing.index, how="left")
                limit = limit.reindex(idx).fillna(0)
                already = existing.reindex(idx).fillna(0)
                if "p_nom" not in already.columns:
                    already["p_nom"] = 0.0
                remaining = (limit - already.p_nom).dropna()
                clipped = remaining < 0
                if clipped.any():
                    logger.info(
                        "%d %s limits are already met by existing capacity in "
                        "%s; new build for those is held at zero.",
                        int(clipped.sum()),
                        component,
                        year,
                    )
                remaining[clipped] = 0
                return xr.DataArray(remaining).rename(dim_0="group")

            minimum = _remaining("min")
            maximum = _remaining("max")
        else:
            minimum = xr.DataArray(component_limits["min"].dropna()).rename(
                dim_0="group"
            )
            maximum = xr.DataArray(component_limits["max"].dropna()).rename(
                dim_0="group"
            )

        index = minimum.indexes["group"].intersection(lhs.indexes["group"])
        if not index.empty:
            n.model.add_constraints(
                lhs.sel(group=index) >= minimum.loc[index],
                name=f"agg_p_nom_min_{component}",
            )
            wrote = True
        index = maximum.indexes["group"].intersection(lhs.indexes["group"])
        if not index.empty:
            n.model.add_constraints(
                lhs.sel(group=index) <= maximum.loc[index],
                name=f"agg_p_nom_max_{component}",
            )
            wrote = True

    if not wrote:
        logger.warning(
            "Country capacity limits found no extendable generators or links."
        )


def _link_country(n: pypsa.Network) -> pd.Series:
    """Country of a link, taken from the first bus that is not an EU bus."""
    country = pd.Series(pd.NA, index=n.links.index, dtype=object)
    bus_columns = sorted(
        (column for column in n.links.columns if re.fullmatch(r"bus\d+", column)),
        key=lambda column: int(column[3:]),
    )
    for column in bus_columns:
        buses = n.links[column]
        valid = buses.notna() & buses.isin(n.buses.index)
        mapped = pd.Series(pd.NA, index=n.links.index, dtype=object)
        if valid.any():
            if "country" in n.buses.columns:
                mapped.loc[valid] = n.buses.country.reindex(buses[valid]).to_numpy()
            missing = valid & mapped.isna()
            if missing.any() and "location" in n.buses.columns:
                mapped.loc[missing] = (
                    n.buses.location.reindex(buses[missing]).to_numpy()
                )
        atmosphere = buses.map(n.buses.carrier).eq("co2")
        mapped = mapped.mask(atmosphere | mapped.isin(["", "EU"]))
        mapped = mapped.mask(mapped.astype(str).str.startswith("EU"))
        take = country.isna() & mapped.notna()
        country.loc[take] = mapped.loc[take]
    return country


def add_national_co2_constraints(
    n: pypsa.Network,
    planning_horizons: str | None,
    snakemake,
) -> None:
    """
    Cap CO2 at a reference inventory times an IEA index.

    The index is ``iea_mt[horizon] / iea_mt[reference_year]``. WEO publishes
    one European Union total. With ``aggregate: eu`` that total becomes one
    cap on the EU countries together. Otherwise each country is capped on its
    own. Emissions are the net flow into the ``co2`` atmosphere bus. Land use
    is not added.
    """
    if planning_horizons is None:
        logger.warning("National CO2 caps are not applied under perfect foresight.")
        return
    settings = n.config.get("co2_budget_national") or {}
    path = snakemake.input.get("co2_totals_reference") if snakemake is not None else None
    if not path:
        raise FileNotFoundError(
            "co2_budget_national is enabled but solve_network did not receive "
            "co2_totals_reference.csv."
        )

    from scripts.prepare_sector_network import determine_emission_sectors

    year = int(planning_horizons)
    reference_year = int(settings["reference_year"])
    fraction = national_co2_fraction(settings["iea_mt"], year, reference_year)
    sectors = determine_emission_sectors(n.config["sector"])
    totals = pd.read_csv(path, index_col=0)
    inventory = totals.reindex(n.config["countries"])[sectors].sum(axis=1)
    aggregate = str(settings.get("aggregate", "country")).lower()
    eu_countries = None
    if aggregate == "eu":
        from scripts.build_iea_cps_capacity_limits import EU_IN_MODEL

        listed = settings.get("countries")
        eu_countries = {str(code) for code in (listed or EU_IN_MODEL)}
        inventory = inventory.reindex(sorted(eu_countries))
    missing = inventory.index[inventory.isna()]
    if len(missing):
        logger.warning(
            "No %s CO2 inventory for %s; those countries are not in the cap.",
            reference_year,
            ", ".join(missing),
        )
    inventory = inventory.dropna()
    nyears = float(n.snapshot_weightings.generators.sum()) / 8760.0
    caps = inventory * 1e6 * fraction * nyears
    if eu_countries is None:
        logger.info(
            "National CO2 caps are %.3f times %s emissions (%.0f Mt over %d countries).",
            fraction,
            reference_year,
            float(inventory.sum() * fraction),
            int(caps.size),
        )
    else:
        logger.info(
            "EU CO2 cap is %.3f times the %s emissions of %d EU countries (%.0f Mt). "
            "Countries outside that group are not in the cap.",
            fraction,
            reference_year,
            int(caps.size),
            float(inventory.sum() * fraction),
        )

    if "Link-p" not in n.model.variables:
        logger.warning("No link flows to constrain with national CO2 caps.")
        return

    country = _link_country(n)
    p = n.model["Link-p"]
    weightings = n.snapshot_weightings.generators
    lhs = None
    unassigned = []
    ports = [
        column[3:]
        for column in n.links.columns
        if re.fullmatch(r"bus\d+", column)
    ]
    for column in ports:
        bus_column = f"bus{column}"
        on_atmosphere = n.links[bus_column].map(n.buses.carrier).eq("co2")
        if column == "0":
            efficiency = pd.Series(-1.0, index=n.links.index)
        elif column == "1":
            efficiency = n.links["efficiency"]
        elif f"efficiency{column}" in n.links.columns:
            efficiency = n.links[f"efficiency{column}"]
        else:
            continue
        idx = n.links.index[on_atmosphere]
        if idx.empty:
            continue
        grouped_country = country.reindex(idx)
        missing_country = grouped_country.isna()
        if missing_country.any():
            unassigned.extend(
                n.links.loc[
                    grouped_country.index[missing_country.to_numpy()], "carrier"
                ].tolist()
            )
            grouped_country = grouped_country.dropna()
        grouped_country = grouped_country.rename("country")
        if eu_countries is not None:
            grouped_country = grouped_country[grouped_country.isin(eu_countries)]
        if grouped_country.empty:
            continue
        kept = grouped_country.index
        weighted = (p.loc[:, kept] * efficiency.reindex(kept).fillna(0)) * weightings
        if eu_countries is None:
            term = weighted.groupby(grouped_country).sum().sum("snapshot")
            lhs = term if lhs is None else lhs + term
        else:
            # One total for the EU. Summing the links directly avoids adding
            # two country-grouped expressions together.
            term = weighted.sum()
            lhs = term if lhs is None else lhs + term

    if unassigned:
        counts = pd.Series(unassigned).value_counts()
        logger.warning(
            "CO2 flows with no country are outside the CO2 cap:\n%s",
            counts.to_string(),
        )
    if lhs is None:
        logger.warning("No country CO2 flows were found to constrain.")
        return

    if eu_countries is not None:
        n.model.add_constraints(
            lhs <= float(caps.sum()),
            name="co2_limit_eu",
        )
        return

    dim = next(iter(lhs.dims))
    constrained = [ct for ct in caps.index if ct in lhs.indexes[dim]]
    skipped = caps.index.difference(constrained)
    if len(skipped):
        logger.info(
            "No emitting links for %s; those national caps are not written.",
            ", ".join(skipped),
        )
    for ct in constrained:
        n.model.add_constraints(
            lhs.sel({dim: ct}) <= float(caps[ct]),
            name=f"co2_limit_country_{ct}",
        )


def add_EQ_constraints(n: pypsa.Network, o: str, scaling: float = 1e-1) -> None:
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators @ n.loads_t.p_set.T.groupby(lgrouper).sum().T
    )
    eff = n.storage_units.efficiency_dispatch
    inflow = n.storage_units_t.inflow
    inflow = (
        n.snapshot_weightings.stores
        @ (inflow * eff[inflow.columns]).T.groupby(sgrouper).sum().T
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling) * eff.to_xarray())
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add business-as-usual (BAU) constraints for minimum capacities.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network instance
    config : dict
        Configuration dictionary containing BAU minimum capacities
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier)
    if not PYPSA_V1:
        ext_carrier_i = ext_carrier_i.rename_axis("Generator-ext")
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance.
    config : dict
        Configuration dictionary.

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(
    n: pypsa.Network, sns: pd.DatetimeIndex, config: dict
) -> None:
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX.jl/stable/Model_Reference/core/#Operational-Reserves.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance.
    sns : pd.DatetimeIndex
        Snapshots for the simulation.
    config : dict
        Configuration dictionary.

    Example
    -------
    config.yaml requires to specify operational_reserve:
    operational_reserve:
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 4000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    vres_carriers = [  # noqa: F841
        "ror" if c == "hydro" else c
        for c in config["electricity"]["renewable_carriers"]
    ]
    generator_dim = "Generator" if not PYPSA_V1 else "name"

    gen_i = n.generators.query("carrier != 'load'").index  # exclude load shedding
    fix_i = n.generators.loc[gen_i].query("not p_nom_extendable").index
    ext_i = n.generators.loc[gen_i].query("p_nom_extendable").index
    vres_i = n.generators.loc[gen_i].query("carrier in @vres_carriers").index

    # Reserve Variables
    n.model.add_variables(0, np.inf, coords=[sns, gen_i], name="Generator-r")
    reserve = n.model["Generator-r"]
    lhs = reserve.sum(generator_dim)

    # Share of extendable renewable capacities
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)]
        if not PYPSA_V1:
            p_nom_vres = p_nom_vres.rename({"Generator-ext": "Generator"})
        lhs += (p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))).sum(
            generator_dim
        )

    # Total demand per t
    demand = n.get_switchable_as_dense("Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded

    dispatch = n.model["Generator-p"].sel(name=gen_i)
    reserve = n.model["Generator-r"]
    lhs = dispatch + reserve

    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")

    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].sel(name=ext_i)
        if not PYPSA_V1:
            capacity_variable = capacity_variable.rename({"Generator-ext": "Generator"})
        lhs -= capacity_variable * xr.DataArray(p_max_pu[ext_i])

    capacity_fixed = n.generators.p_nom[fix_i]
    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_TES_energy_to_power_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES constraints to the network.

    For each TES storage unit, enforce:
        Store-e_nom - etpr * Link-p_nom == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES storage or charger links are found.
    RuntimeError
        If the TES storage and charger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains("water tanks charger|water pits charger")
        & n.links.p_nom_extendable
    ]
    indices_stores_e_nom_extendable = n.stores.index[
        n.stores.index.str.contains("water tanks|water pits")
        & n.stores.e_nom_extendable
    ]

    if indices_charger_p_nom_extendable.empty or indices_stores_e_nom_extendable.empty:
        logger.debug(
            "No valid extendable charger links or stores found for TES energy-to-power constraints.Not enforcing TES energy-to-power ratio constraints!"
        )
        return

    energy_to_power_ratio_values = n.links.loc[
        indices_charger_p_nom_extendable, "energy to power ratio"
    ].values

    linear_expr_list = []
    for charger, tes, energy_to_power_value in zip(
        indices_charger_p_nom_extendable,
        indices_stores_e_nom_extendable,
        energy_to_power_ratio_values,
        strict=True,
    ):
        charger_var = n.model["Link-p_nom"].loc[charger]
        if not tes == charger.replace(" charger", ""):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and TES {tes} do not match. "
                "Ensure that the charger and TES are in the same location and refer to the same technology."
            )
        store_var = n.model["Store-e_nom"].loc[tes]
        linear_expr = store_var - energy_to_power_value * charger_var
        linear_expr_list.append(linear_expr)

    # Merge the individual expressions
    dim = "Store-ext, Link-ext" if PYPSA_V1 else "name"
    merged_expr = linopy.expressions.merge(
        linear_expr_list, dim=dim, cls=type(linear_expr_list[0])
    )

    n.model.add_constraints(merged_expr == 0, name="TES_energy_to_power_ratio")


def add_TES_charger_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES charger ratio constraints.

    For each TES unit, enforce:
        Link-p_nom(charger) - efficiency * Link-p_nom(discharger) == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES discharger or charger links are found.
    RuntimeError
        If the charger and discharger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks charger|water pits charger|aquifer thermal energy storage charger"
        )
        & n.links.p_nom_extendable
    ]
    indices_discharger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks discharger|water pits discharger|aquifer thermal energy storage discharger"
        )
        & n.links.p_nom_extendable
    ]

    if (
        indices_charger_p_nom_extendable.empty
        or indices_discharger_p_nom_extendable.empty
    ):
        logger.debug(
            "No valid extendable TES discharger or charger links found for TES charger ratio constraints. Not enforcing TES charger_ratio constraints."
        )
        return

    for charger, discharger in zip(
        indices_charger_p_nom_extendable,
        indices_discharger_p_nom_extendable,
        strict=True,
    ):
        if not charger.replace(" charger", " ") == discharger.replace(
            " discharger", " "
        ):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and discharger {discharger} do not match. "
                "Ensure that the charger and discharger are in the same location and refer to the same technology."
            )

    eff_discharger = n.links.efficiency[indices_discharger_p_nom_extendable].values
    lhs = (
        n.model["Link-p_nom"].loc[indices_charger_p_nom_extendable]
        - n.model["Link-p_nom"].loc[indices_discharger_p_nom_extendable]
        * eff_discharger
    )

    n.model.add_constraints(lhs == 0, name="TES_charger_ratio")


def add_battery_constraints(n: pypsa.Network) -> None:
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n: pypsa.Network) -> None:
    if not n.links.p_nom_extendable.any() or not any(n.links.get("reversed", [])):
        return

    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841
    backwards = n.links.query(
        "carrier in @carriers and p_nom_extendable and reversed and active"
    ).index
    forwards = backwards.str.replace("-reversed", "")
    lhs = n.model["Link-p_nom"].loc[backwards]
    rhs = n.model["Link-p_nom"].loc[forwards]
    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n: pypsa.Network) -> None:
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {} if PYPSA_V1 else {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n: pypsa.Network) -> None:
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed and active"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed and active"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    config = n.config
    CH4_per_H2 = 1 / config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i]
    if not PYPSA_V1:
        rhs = rhs.rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n: pypsa.Network) -> None:
    """
    Limit each geothermal reservoir to the capacity of the wells at its surface bus.
    """
    wells = n.links.query("carrier == 'geothermal heat'")
    new_wells = wells.query("p_nom_extendable")
    reservoirs = n.storage_units.query(
        "carrier == 'geothermal heat' and p_nom_extendable"
    )
    if not PYPSA_V1:
        new_wells = new_wells.rename_axis("Link-ext")
        reservoirs = reservoirs.rename_axis("StorageUnit-ext")

    reservoir_p_nom = n.model["StorageUnit-p_nom"].loc[reservoirs.index]
    well_p_nom = n.model["Link-p_nom"].loc[new_wells.index]
    lhs = reservoir_p_nom.groupby(reservoirs.bus).sum()
    rhs = well_p_nom.groupby(new_wells.bus1.rename("bus")).sum()
    existing = (
        wells.query("not p_nom_extendable")
        .groupby("bus1")
        .p_nom.sum()
        .reindex(reservoirs.bus.unique(), fill_value=0.0)
        .rename_axis("bus")
    )

    n.model.add_constraints(
        lhs - rhs <= existing.to_xarray(),
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_import_limit_constraint(n: pypsa.Network, sns: pd.DatetimeIndex) -> None:
    """
    Add constraint for limiting green energy imports (synthetic and biomass).
    Does not include fossil fuel imports.
    """

    nyears = n.snapshot_weightings.generators.sum() / 8760

    import_links = n.links.loc[n.links.carrier.str.contains("import")].index
    import_gens = n.generators.loc[n.generators.carrier.str.contains("import")].index

    config = n.config
    limit = config["sector"]["imports"]["limit"]
    limit_sense = config["sector"]["imports"]["limit_sense"]

    if (import_links.empty and import_gens.empty) or not np.isfinite(limit):
        return

    weightings = n.snapshot_weightings.loc[sns, "generators"]

    # everything needs to be in MWh_fuel
    eff = n.links.loc[import_links, "efficiency"]

    p_gens = n.model["Generator-p"].loc[sns, import_gens]
    p_links = n.model["Link-p"].loc[sns, import_links]

    lhs = (p_gens * weightings).sum() + (p_links * eff * weightings).sum()

    rhs = limit * 1e6 * nyears

    n.model.add_constraints(lhs, limit_sense, rhs, name="import_limit")


def add_co2_atmosphere_constraint(
    n: pypsa.Network, snapshots: pd.DatetimeIndex
) -> None:
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def extra_functionality(
    n: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    planning_horizons: str | None = None,
    snakemake=None,
) -> None:
    """
    Add custom constraints and functionality.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance with config and params attributes
    snapshots : pd.DatetimeIndex
        Simulation timesteps
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight.

    Notes
    -----
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and (
        n.generators.p_nom_extendable.any() or n.links.p_nom_extendable.any()
    ):
        add_CCL_constraints(n, config, planning_horizons)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        if config["sector"]["enabled"]:
            logger.warning(
                "Operational reserve margin only considers electricity generators "
                "and ignores sector-coupling components."
            )
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))

    if {"solar-hsat", "solar"}.issubset(
        config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    add_TES_energy_to_power_ratio_constraints(n)
    add_TES_charger_ratio_constraints(n)
    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)

    egs = config["sector"]["enhanced_geothermal"]
    if egs["enable"] and egs["flexible"]:
        add_flexible_egs_constraint(n)

    if config["sector"]["imports"]["enable"]:
        add_import_limit_constraint(n, snapshots)

    national_co2 = config.get("co2_budget_national") or {}
    if national_co2.get("enable"):
        add_national_co2_constraints(n, planning_horizons, snakemake)

    if n.params.custom_extra_functionality:
        source_path = n.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def check_objective_value(n: pypsa.Network, solving: dict, horizon: str) -> None:
    """
    Check if objective value matches expected value within tolerance.

    Parameters
    ----------
    n : pypsa.Network
        Network with solved objective
    solving : Dict
        Dictionary containing objective checking parameters
    horizon : str
        Planning horizon of the current solve, used to select the expected
        value for myopic foresight (per-horizon mapping)

    Raises
    ------
    ObjectiveValueError
        If objective value differs from expected value beyond tolerance
    """
    check_objective = solving["check_objective"]
    if not check_objective["enable"]:
        return
    atol = check_objective["atol"]
    rtol = check_objective["rtol"]
    expected_value = check_objective["expected_value"]
    if isinstance(expected_value, dict):
        expected_value = expected_value[int(horizon)]
    if not np.isclose(n.objective, expected_value, atol=atol, rtol=rtol):
        raise ObjectiveValueError(
            f"Objective value {n.objective} differs from expected value "
            f"{expected_value} by more than {atol}."
        )


def collect_kwargs(
    config: dict,
    solving: dict,
    planning_horizons: str | None = None,
    log_fn: str | None = None,
    mode: str = "single",
    horizon: int | None = None,
    overlap: int | None = None,
) -> tuple[dict, dict]:
    """
    Prepare keyword arguments separated for model creation and model solving.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing solver settings
    solving : dict
        Dictionary of solving options and configuration
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    log_fn : str, optional
        Path to solver log file
    mode : str, optional
        Optimization mode: 'single', 'rolling_horizon', or 'iterative'
        Default is 'single'

    Returns
    -------
    tuple[dict, dict]
        Two dictionaries: (model_kwargs, solve_kwargs)
        - model_kwargs: Arguments for n.optimize.create_model()
        - solve_kwargs: Arguments for n.optimize.solve_model()
        For 'rolling_horizon' and 'iterative' modes, returns merged kwargs
        with additional mode-specific parameters
    """
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    # Model creation kwargs
    model_kwargs = {}
    model_kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    model_kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    model_kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )

    # Solve kwargs
    solver_name = solving["solver"]["name"]
    solver_options = solving["solver_options"][set_of_options] if set_of_options else {}

    solve_kwargs = {}
    solve_kwargs["solver_name"] = solver_name
    solve_kwargs["solver_options"] = solver_options
    solve_kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    solve_kwargs["io_api"] = cf_solving.get("io_api", None)
    solve_kwargs["keep_files"] = cf_solving.get("keep_files", False)

    if log_fn:
        solve_kwargs["log_fn"] = log_fn

    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    # Handle special modes
    if mode == "rolling_horizon":
        all_kwargs = {**model_kwargs, **solve_kwargs}
        all_kwargs["horizon"] = (
            horizon if horizon is not None else cf_solving["horizon"]
        )
        all_kwargs["overlap"] = (
            overlap if overlap is not None else cf_solving["overlap"]
        )
        return all_kwargs, {}

    elif mode == "iterative":
        all_kwargs = {**model_kwargs, **solve_kwargs}
        all_kwargs["track_iterations"] = cf_solving["track_iterations"]
        all_kwargs["min_iterations"] = cf_solving["min_iterations"]
        all_kwargs["max_iterations"] = cf_solving["max_iterations"]

        if cf_solving["post_discretization"].get("enable", False):
            logger.info("Add post-discretization parameters.")
            cf_solving["post_discretization"].pop("enable", None)
            all_kwargs.update(cf_solving["post_discretization"])

        return all_kwargs, {}

    return model_kwargs, solve_kwargs


def create_optimization_model(
    n: pypsa.Network,
    config: dict,
    params: dict,
    model_kwargs: dict,
    solve_kwargs: dict,
    planning_horizons: str | None = None,
    snakemake=None,
) -> None:
    """
    Prepare optimization problem by creating model and adding extra functionality.

    This function:
    1. Attaches config and params to network for extra_functionality
    2. Creates the optimization model
    3. Adds extra functionality (custom constraints)

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary containing solver settings
    params : dict
        Dictionary of solving parameters
    model_kwargs : dict
        Arguments for n.optimize.create_model()
    solve_kwargs : dict
        Arguments for n.optimize.solve_model()
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """
    # Add config and params to network for extra_functionality
    n.config = config
    n.params = params

    # Create optimization model
    logger.info("Creating optimization model...")
    n.optimize.create_model(**model_kwargs)

    # Add extra functionality (custom constraints)
    logger.info("Adding extra functionality (custom constraints)...")
    extra_functionality(n, n.snapshots, planning_horizons, snakemake=snakemake)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            configfiles="config/test/config.overnight.yaml",
            horizon=2030,
        )
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    solve_opts = snakemake.params.solving["options"]
    cf_solving = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    # Load network
    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.horizon

    # Prepare network (settings before solving)
    prepare_network(
        n,
        config=snakemake.config,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params["sector"]["limit_max_growth"],
        rolling_horizon=cf_solving["rolling_horizon"],
    )

    # Determine solve mode
    rolling_horizon = cf_solving["rolling_horizon"]
    skip_iterations = cf_solving.get("skip_iterations", False)

    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )

    # Solve network based on mode
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        if rolling_horizon:
            logger.info("Using rolling horizon optimization...")
            all_kwargs, _ = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="rolling_horizon",
            )

            n.config = snakemake.config
            n.params = snakemake.params
            all_kwargs["extra_functionality"] = partial(
                extra_functionality,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )
            n.optimize.optimize_with_rolling_horizon(**all_kwargs)
            status, condition = "", ""

        elif skip_iterations:
            logger.info("Using single-pass optimization...")
            model_kwargs, solve_kwargs = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="single",
            )
            create_optimization_model(
                n,
                config=snakemake.config,
                params=snakemake.params,
                model_kwargs=model_kwargs,
                solve_kwargs=solve_kwargs,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )

            logger.info("Solving model...")
            status, condition = n.optimize.solve_model(**solve_kwargs)

        else:
            logger.info("Using iterative transmission expansion optimization...")

            all_kwargs, _ = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="iterative",
            )

            n.config = snakemake.config
            n.params = snakemake.params
            all_kwargs["extra_functionality"] = partial(
                extra_functionality,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )
            status, condition = n.optimize.optimize_transmission_expansion_iteratively(
                **all_kwargs
            )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    # Check results
    if not rolling_horizon:
        if status != SolverStatus.ok:
            logger.warning(
                f"Solving status '{status}' with termination condition '{condition}'"
            )
        check_objective_value(n, snakemake.params.solving, snakemake.wildcards.horizon)

    if condition in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasible_or_unbounded,
    ]:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'. Infeasibilities computed.")

    if status == SolverStatus.warning:
        raise RuntimeError("Solving status 'warning'. Discarding solution.")

    n.meta["snakemake_config"] = dict(
        snakemake.config, wildcards=dict(snakemake.wildcards)
    )
    n.export_to_netcdf(snakemake.output.network)

    if snakemake.output.get("model"):
        n.model.to_netcdf(snakemake.output.model)

    config_output = getattr(snakemake.output, "config", None)
    if config_output:
        with open(config_output, "w") as file:
            yaml.dump(
                n.meta["snakemake_config"],
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
