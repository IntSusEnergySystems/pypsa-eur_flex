# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Compose the network for one planning horizon from the electricity and
sector-coupling building blocks.

Starting from the clustered base network, the script adds the electricity
components (`add_electricity`), the sector-coupling components if enabled
(`prepare_sector_network`), applies temporal aggregation and the final
preparation for solving (`prepare_network`), and then handles the planning
horizon according to the foresight mode. In the first horizon of a myopic or
perfect foresight run, existing capacities are added (`add_existing_baseyear`).
In later myopic horizons, the optimised capacities of the previous horizon's
solved network are carried over as brownfield (`add_brownfield`). Under
perfect foresight, the horizon is stacked onto the previous composed network
to form a multi-period network (`prepare_perfect_foresight`). Finally, CO2
budget constraints and configured adjustments to costs and potentials are
applied and the network is exported for solving.
"""

import logging
import re

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    load_costs,
    sanitize_custom_columns,
    set_scenario_config,
)
from scripts.add_brownfield import adjust_renewable_capacity_limits
from scripts.add_brownfield import main as apply_brownfield
from scripts.add_electricity import main as add_electricity_components
from scripts.add_electricity import (
    sanitize_carriers,
    sanitize_locations,
)
from scripts.add_existing_baseyear import main as add_existing_capacities
from scripts.build_iea_cps_capacity_limits import carrier_limit
from scripts.nuclear_fleet import apply_nuclear_capacity
from scripts.prepare_network import (
    apply_co2_budget_constraints,
    apply_temporal_aggregation,
    maybe_adjust_costs_and_potentials,
)
from scripts.prepare_network import main as prepare_network_for_solving
from scripts.prepare_perfect_foresight import apply_phase_outs
from scripts.prepare_perfect_foresight import main as prepare_perfect_foresight
from scripts.prepare_sector_network import (
    main as add_sector_components,
)

logger = logging.getLogger(__name__)

SECTOR_RENEWABLE_CARRIERS = {"solar rooftop", "afforestation", "biochar", "perennials"}
_COUNTRY_AC = re.compile(r"([A-Z]{2})AC")
_BUS_COLUMNS = ("bus", "bus0", "bus1", "bus2", "bus3", "bus4", "location")
_SUM_COLUMNS = {
    "p_nom",
    "p_nom_min",
    "p_nom_max",
    "e_nom",
    "e_nom_min",
    "e_nom_max",
    "e_sum_min",
    "e_sum_max",
    "e_initial",
    "p_set",
    "inflow",
}
_WEIGHT_COLUMNS = {
    "marginal_cost",
    "capital_cost",
    "efficiency",
    "efficiency2",
    "efficiency3",
    "efficiency4",
    "standing_loss",
    "p_max_pu",
    "p_min_pu",
    "marginal_cost_quadratic",
}
_SUM_DYNAMIC = {"p_set", "inflow"}
_COMPONENTS = ("Generator", "Load", "Link", "Line", "Store", "StorageUnit", "Bus")


def _country_name(name: object) -> str:
    """ATAC and ATAC gas become AT and AT gas. Northern Ireland joins GB."""
    text = str(name).replace("GBIAC", "GB")
    return _COUNTRY_AC.sub(r"\1", text)


def _component(n: pypsa.Network, name: str):
    try:
        block = n.components[name]
    except KeyError:
        return None
    if block.static.empty:
        return None
    return block


def _preferred(names: list[str]) -> str:
    return sorted(names, key=lambda name: ("GBI" in str(name), str(name)))[0]


def _sum_numbers(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows[column], errors="coerce")
    array = values.to_numpy(dtype=float)
    # A minimum of -inf means "no minimum". Do not turn that into +inf.
    if column.endswith("_min"):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return -np.inf if np.isneginf(array).any() else np.nan
        return float(finite.sum())
    if np.isposinf(array).any():
        return np.inf
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan
    return float(finite.sum())


def _profile_weight(rows: pd.DataFrame) -> pd.Series:
    if "p_nom_max" in rows.columns:
        weight = pd.to_numeric(rows["p_nom_max"], errors="coerce")
        weight = weight.where(np.isfinite(weight), 0.0)
        if float(weight.sum()) > 0:
            return weight
    for column in ("p_nom", "e_nom", "p_set"):
        if column not in rows.columns:
            continue
        weight = pd.to_numeric(rows[column], errors="coerce")
        weight = weight.where(np.isfinite(weight), 0.0).fillna(0.0)
        if float(weight.sum()) > 0:
            return weight
    return pd.Series(1.0, index=rows.index)


def _weighted_mean(rows: pd.DataFrame, column: str, weight: pd.Series) -> float:
    values = pd.to_numeric(rows[column], errors="coerce")
    finite = values.where(np.isfinite(values))
    used = weight.reindex(rows.index).where(np.isfinite(weight), 0.0).fillna(0.0)
    used = used.where(finite.notna(), 0.0)
    if float(used.sum()) <= 0:
        return float(finite.mean()) if finite.notna().any() else np.nan
    return float((finite.fillna(0.0) * used).sum() / float(used.sum()))


def _rewrite_bus_columns(frame: pd.DataFrame) -> None:
    for column in _BUS_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = frame[column].map(
            lambda value: value if pd.isna(value) else _country_name(value)
        )


def _collapse_static(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for name in frame.index:
        groups.setdefault(_country_name(name), []).append(name)
    if all(len(members) == 1 for members in groups.values()):
        renamed = frame.copy()
        renamed.index = renamed.index.map(_country_name)
        renamed.index.name = frame.index.name
        return renamed

    rows = []
    for new_name, members in groups.items():
        if len(members) == 1:
            row = frame.loc[members[0]].copy()
            row.name = new_name
            rows.append(row)
            continue
        chosen = frame.loc[members]
        if component == "Bus":
            row = frame.loc[_preferred(members)].copy()
            row.name = new_name
            rows.append(row)
            continue
        row = frame.loc[_preferred(members)].copy()
        weight = _profile_weight(chosen)
        for column in chosen.columns:
            if column in _SUM_COLUMNS:
                row[column] = _sum_numbers(chosen, column)
            elif column in _WEIGHT_COLUMNS and pd.api.types.is_numeric_dtype(chosen[column]):
                row[column] = _weighted_mean(chosen, column, weight)
            elif pd.api.types.is_bool_dtype(chosen[column]):
                row[column] = bool(chosen[column].fillna(False).any())
        row.name = new_name
        rows.append(row)
    collapsed = pd.DataFrame(rows)
    collapsed.index.name = frame.index.name
    return collapsed


def _collapse_dynamic(frame: pd.DataFrame, how: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    groups: dict[str, list[str]] = {}
    for name in frame.columns:
        groups.setdefault(_country_name(name), []).append(name)
    pieces = []
    for new_name, members in groups.items():
        present = [member for member in members if member in frame.columns]
        if not present:
            continue
        if len(present) == 1 or how == "sum":
            series = frame[present].sum(axis=1) if how == "sum" and len(present) > 1 else frame[present[0]]
        else:
            series = frame[present].mean(axis=1)
        series.name = new_name
        pieces.append(series)
    if not pieces:
        return frame.iloc[:, 0:0]
    return pd.concat(pieces, axis=1)


def use_country_names(n: pypsa.Network) -> None:
    """Name buses and components by country, and fold Northern Ireland into GB."""
    for component in _COMPONENTS:
        block = _component(n, component)
        if block is None:
            continue
        _rewrite_bus_columns(block.static)

    if not n.lines.empty:
        internal = n.lines.index[n.lines.bus0 == n.lines.bus1]
        if len(internal):
            n.remove("Line", internal)
            logger.info(
                "Removed %d AC lines inside a country after Northern Ireland joined GB.",
                len(internal),
            )
    if not n.links.empty and {"bus0", "bus1"} <= set(n.links.columns):
        internal = n.links.index[n.links.bus0.notna() & (n.links.bus0 == n.links.bus1)]
        if len(internal):
            n.remove("Link", internal)
            logger.info("Removed %d links that ran inside one bus.", len(internal))

    for component in _COMPONENTS:
        block = _component(n, component)
        if block is None:
            continue
        static = _collapse_static(block.static, component)
        # Availability of merged plants is weighted below; other series are summed
        # when they are demands and averaged when the static pass already merged them.
        dynamic = {}
        for attr, values in block.dynamic.items():
            if values is None or values.empty:
                dynamic[attr] = values
                continue
            how = "sum" if attr in _SUM_DYNAMIC else "mean"
            if how == "sum":
                dynamic[attr] = _collapse_dynamic(values, how)
            elif component in {"Generator", "Link", "Store", "StorageUnit"}:
                dynamic[attr] = _weighted_series(block.static, values)
            else:
                dynamic[attr] = _collapse_dynamic(values, "mean")
        block.static = static
        for attr, values in dynamic.items():
            if values is None or getattr(values, "empty", True):
                block.dynamic[attr] = values
                continue
            keep = [name for name in static.index if name in values.columns]
            block.dynamic[attr] = values.loc[:, keep]

    if "country" in n.buses.columns and "location" in n.buses.columns:
        country = n.buses["country"].astype(str)
        named = country.str.fullmatch(r"[A-Z]{2}")
        n.buses.loc[named.fillna(False), "location"] = country[named.fillna(False)]
    logger.info(
        "Buses, locations and component names use the country code. "
        "Northern Ireland is part of GB."
    )


def _weighted_series(static: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted availability when two plants become one."""
    groups: dict[str, list[str]] = {}
    for name in static.index:
        groups.setdefault(_country_name(name), []).append(name)
    pieces = []
    for new_name, members in groups.items():
        present = [member for member in members if member in profile.columns]
        if not present:
            continue
        if len(present) == 1:
            series = profile[present[0]]
        else:
            weight = _profile_weight(static.loc[present])
            used = weight.reindex(present).fillna(0.0)
            if float(used.sum()) <= 0:
                series = profile[present].mean(axis=1)
            else:
                series = profile[present].mul(used, axis=1).sum(axis=1) / float(used.sum())
        series.name = new_name
        pieces.append(series)
    if not pieces:
        return profile.iloc[:, 0:0]
    return pd.concat(pieces, axis=1)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "compose_network",
            run="test-run",
            horizon="2050",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Extract configuration and parameters
    params = snakemake.params
    inputs = snakemake.input

    current_horizon = int(snakemake.wildcards.horizon)
    horizons = params.horizons
    foresight = params.foresight
    sector_mode = params.sector["enabled"]

    is_first_horizon = current_horizon == horizons[0]

    phase_outs = params.existing_capacities["phase_outs"]
    if phase_outs and foresight != "perfect":
        logger.warning(
            "existing_capacities.phase_outs is only applied under perfect foresight; "
            f"the {len(phase_outs)} configured rule(s) are ignored for foresight "
            f"'{foresight}', where phase-outs follow the powerplant lifetimes instead."
        )

    n = pypsa.Network(snakemake.input.network)
    n_previous = None if is_first_horizon else pypsa.Network(inputs.network_previous)
    nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(snakemake.input.tech_costs)

    add_electricity_components(n, inputs, params, costs)

    if sector_mode:
        add_sector_components(n, inputs, params, costs, nyears, current_horizon)

    apply_temporal_aggregation(n, inputs, params)

    renewable_carriers = sorted(
        set(params.renewable_carriers)
        | (
            SECTOR_RENEWABLE_CARRIERS
            & (set(n.generators.carrier) | set(n.links.carrier))
        )
    )

    if foresight != "overnight" and is_first_horizon:
        add_existing_capacities(n, inputs, params, costs)
        if foresight == "myopic":
            adjust_renewable_capacity_limits(
                n, str(current_horizon), renewable_carriers
            )

    prepare_network_for_solving(n, inputs, params, costs, nyears, current_horizon)

    if foresight == "myopic" and not is_first_horizon:
        apply_brownfield(
            n, n_previous, inputs, params, current_horizon, renewable_carriers
        )

    if params.electricity.get("nuclear_fleet", False) and foresight == "myopic":
        # One generator per country. EU countries that have a Current Policies
        # share are held at that share. The minimum stays at the fleet already
        # in service, so capacity above it keeps its annualised capital cost.
        powerplants = inputs.powerplants
        if not isinstance(powerplants, str):
            powerplants = list(powerplants)[0]
        cps_max = None
        agg_file = (
            snakemake.config.get("solving", {}).get("agg_p_nom_limits", {}).get("file")
        )
        if agg_file:
            cps_max = carrier_limit(agg_file, "nuclear", current_horizon)
            if cps_max.empty:
                cps_max = None
        apply_nuclear_capacity(
            n,
            powerplants,
            current_horizon,
            n_previous=None if is_first_horizon else n_previous,
            cps_max=cps_max,
        )

    if foresight == "perfect":
        n = prepare_perfect_foresight(n, n_previous, params, current_horizon)
        apply_phase_outs(n, phase_outs, horizons)

    apply_co2_budget_constraints(
        n, inputs=inputs, params=params, nyears=nyears, current_horizon=current_horizon
    )

    adjustments = params.adjustments
    maybe_adjust_costs_and_potentials(n, adjustments["electricity"], current_horizon)
    maybe_adjust_costs_and_potentials(n, adjustments["sector"], current_horizon)

    sanitize_custom_columns(n)
    sanitize_carriers(n, params.nice_names, params.tech_colors)
    use_country_names(n)
    sanitize_locations(n)
    n.meta["snakemake_config"] = dict(
        snakemake.config, wildcards=dict(snakemake.wildcards)
    )
    if n.investment_periods.empty:
        n.consistency_check()
    else:
        logger.warning(
            "Skipping consistency check for multi-period (perfect foresight) network: "
            "known pandas/PyPSA MultiIndex limitation."
        )

    logger.info(f"Exporting composed network for horizon {current_horizon}")
    n.export_to_netcdf(snakemake.output[0])
