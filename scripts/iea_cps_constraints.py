# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Impose the WEO 2019 Current Policies totals that this model can represent.

Capacity above a country share is derated. Concentrating solar and marine
are must-run plants with no cost. Combustion CO2 and final energy use the
published EU totals. Primary oil, gas and coal are only reported.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# 1 toe = 41.868 GJ. 1 Mtoe = 11.63 million MWh.
MWH_PER_MTOE = 41.868e6 / 3.6
# 7.33 barrels per tonne of oil equivalent.
MBD_TO_MWH = 1e6 * 365.25 / 7.33 * (41.868 / 3.6)
# Table A.3 gas is 398 Mtoe against Table A.1 demand of 485 bcm.
BCM_TO_MWH = (398.0 / 485.0) * MWH_PER_MTOE
# 1 tonne of coal equivalent = 0.7 toe. 239 Mtce is the 167 Mtoe coal row.
MTCE_TO_MWH = 0.7 * MWH_PER_MTOE

LOAD_GROUPS = {
    ("industry", "coal"): ["coal for industry"],
    ("industry", "gas"): ["gas for industry"],
    ("industry", "electricity"): ["industry electricity"],
    ("industry", "heat"): ["low-temperature heat for industry"],
    ("industry", "bioenergy"): ["solid biomass for industry"],
    ("industry", "oil"): ["naphtha for industry"],
    ("transport", "oil"): [
        "land transport oil",
        "kerosene for aviation",
        "shipping oil",
    ],
    ("transport", "electricity"): ["land transport EV"],
    ("buildings", "electricity"): ["electricity"],
    ("buildings", "heat"): ["urban central heat"],
}


def published_value(pair: dict, year: int) -> float:
    """Return the published value, with 2035 the geometric midpoint."""
    from scripts.solve_network import _iea_extend

    def _get(key: int) -> float:
        if key in pair:
            return float(pair[key])
        return float(pair[str(key)])

    return _iea_extend(_get(2030), _get(2040))[year]


def _nyears(n: pypsa.Network) -> float:
    return float(n.snapshot_weightings.generators.sum()) / 8760.0


def _ac_bus(n: pypsa.Network, country: str) -> str | None:
    buses = n.buses.index[n.buses.carrier.eq("AC")]
    if "country" in n.buses.columns:
        buses = buses[n.buses.loc[buses, "country"].astype(str).eq(country)]
    if len(buses) == 0:
        return None
    return str(buses[0])


def derate_existing_to_cps(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Lower in-service capacity that exceeds its Current Policies country share.

    The share is the ``max`` column of the aggregate capacity file, in
    electrical MW. Fuel-side link capacity is reduced in proportion.
    """
    if planning_horizons is None:
        return
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    if not options.get("file"):
        return
    from scripts.solve_network import _ccl_horizon_block, _ccl_rename

    table = pd.read_csv(options["file"], index_col=[0, 1], header=[0, 1])
    limits = _ccl_horizon_block(table, planning_horizons)
    rename = _ccl_rename(limits, options)
    year = int(planning_horizons)
    capped = limits["max"].dropna()
    capped = capped[capped >= 0]
    solar_level = capped.index.get_level_values("carrier")
    solar_caps = capped[solar_level == "solar-all"]
    eu_solar = float(solar_caps.sum()) if len(solar_caps) else 0.0
    removed = 0.0
    groups = 0

    for component, bus_column in (("Generator", "bus"), ("Link", "bus1")):
        static = n.generators if component == "Generator" else n.links
        if static.empty or bus_column not in static.columns:
            continue
        assets = static.copy()
        if {"build_year", "lifetime"} <= set(assets.columns):
            online = assets["build_year"] + assets["lifetime"] >= year
            undated = assets["build_year"].isna() | assets["lifetime"].isna()
            assets = assets.loc[online | undated]
        country = assets[bus_column].map(n.buses.country)
        carrier = assets["carrier"].replace(rename)
        electrical = assets["p_nom"].fillna(0.0)
        if component == "Link":
            electrical = electrical * assets["efficiency"].fillna(1.0)
        frame = pd.DataFrame(
            {"country": country, "carrier": carrier, "electrical": electrical}
        )
        frame = frame.dropna(subset=["country"])
        frame = frame.loc[frame["electrical"] > 0]
        if frame.empty:
            continue
        for (code, agg), block in frame.groupby(["country", "carrier"]):
            key = (code, agg)
            if key not in capped.index:
                continue
            target = float(capped.loc[key])
            if agg == "solar-all":
                target = max(0.0, target - _csp_reservation_mw(config, year, target, eu_solar))
            have = float(block["electrical"].sum())
            if have <= target * (1 + 1e-6):
                continue
            scale = 0.0 if target <= 0 else target / have
            idx = block.index
            static.loc[idx, "p_nom"] = static.loc[idx, "p_nom"].fillna(0.0) * scale
            if "p_nom_min" in static.columns:
                static.loc[idx, "p_nom_min"] = (
                    static.loc[idx, "p_nom_min"].fillna(0.0) * scale
                )
                static.loc[idx, "p_nom_min"] = static.loc[idx, ["p_nom_min", "p_nom"]].min(
                    axis=1
                )
            if "p_nom_extendable" in static.columns and "p_nom_max" in static.columns:
                extendable = static.loc[idx, "p_nom_extendable"].fillna(False).astype(bool)
                chosen = idx[extendable.to_numpy()]
                if len(chosen):
                    static.loc[chosen, "p_nom_max"] = static.loc[chosen, "p_nom"]
            removed += have - target
            groups += 1
    if groups:
        logger.info(
            "Derated %d country technologies by %.1f GW so capacity sits on the "
            "Current Policies share.",
            groups,
            removed / 1000.0,
        )


def _csp_reservation_mw(
    config: dict, year: int, country_mw: float, eu_mw: float
) -> float:
    """Electrical MW of must-run CSP that shares this country's solar limit."""
    spec = config.get("iea_must_run") or {}
    pair = (spec.get("csp") or {}).get("gw") if spec.get("enable") else None
    if not pair or eu_mw <= 0:
        return 0.0
    return published_value(pair, year) * 1000.0 * country_mw / eu_mw


def add_csp_and_marine(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add must-run concentrating solar and marine at the published GW and TWh.

    The outlook gives no cost, so capital and marginal cost stay at zero.
    Country shares follow solar capacity in the aggregate file.
    """
    spec = config.get("iea_must_run") or {}
    if not spec.get("enable") or planning_horizons is None:
        return
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    year = int(planning_horizons)
    shares = _solar_shares(n, options.get("file"), year)
    if shares.empty:
        logger.warning("No AC bus is available for concentrating solar and marine.")
        return
    for carrier, pair in spec.items():
        if carrier == "enable" or not isinstance(pair, dict):
            continue
        gw = published_value(pair.get("gw", pair), year)
        twh = published_value(pair.get("twh", pair), year)
        if gw <= 0:
            continue
        cf = twh / (gw * 8.76)
        if cf > 1:
            logger.warning(
                "%s needs a capacity factor of %.2f. It is held at 1.",
                carrier,
                cf,
            )
            cf = 1.0
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
        for code, share in shares.items():
            if share <= 0:
                continue
            bus = _ac_bus(n, str(code))
            if bus is None:
                continue
            name = f"{code} {carrier}"
            if name in n.generators.index:
                continue
            n.add(
                "Generator",
                name,
                bus=bus,
                carrier=carrier,
                p_nom=float(share) * gw * 1000.0,
                p_nom_extendable=False,
                p_min_pu=cf,
                p_max_pu=cf,
                capital_cost=0.0,
                marginal_cost=0.0,
            )
        logger.info(
            "%s is must-run at %.1f GW and %.1f TWh, with no cost in the objective.",
            carrier,
            gw,
            twh,
        )


def _solar_shares(n: pypsa.Network, path: str | None, year: int) -> pd.Series:
    countries = []
    if "country" in n.buses.columns:
        ac = n.buses.loc[n.buses.carrier.eq("AC"), "country"].dropna().astype(str)
        countries = sorted(set(ac) - {"", "EU"})
    shares = pd.Series(0.0, index=countries, dtype=float)
    if path:
        from scripts.solve_network import _ccl_horizon_block

        table = pd.read_csv(path, index_col=[0, 1], header=[0, 1])
        block = _ccl_horizon_block(table, year)
        solar = block.xs("solar-all", level="carrier")["max"].dropna()
        solar = solar.reindex(shares.index).fillna(0.0)
        total = float(solar.sum())
        if total > 0:
            return solar / total
    if len(shares):
        shares[:] = 1.0 / len(shares)
    return shares


def _row_mtoe(spec: dict, sector: str, fuel: str, year: int) -> float:
    pair = (spec.get(sector) or {}).get(fuel)
    if not pair:
        return 0.0
    return published_value(pair, year)


def _load_index(n: pypsa.Network, carriers: list[str]) -> pd.Index:
    if n.loads.empty:
        return pd.Index([])
    return n.loads.index[n.loads.carrier.isin(carriers)]


def _load_energy_mwh(n: pypsa.Network, loads: pd.Index) -> float:
    weight = n.snapshot_weightings.generators
    total = 0.0
    timed = n.loads_t.p_set
    for load in loads:
        if load in timed.columns and timed[load].notna().any():
            total += float((timed[load].fillna(0.0) * weight).sum())
        else:
            total += float(n.loads.at[load, "p_set"]) * float(weight.sum())
    return total


def _scale_loads(n: pypsa.Network, loads: pd.Index, target_mwh: float) -> bool:
    current = _load_energy_mwh(n, loads)
    if current <= 0:
        return False
    factor = target_mwh / current
    timed = n.loads_t.p_set
    for load in loads:
        if load in timed.columns and timed[load].notna().any():
            n.loads_t.p_set[load] = timed[load].fillna(0.0) * factor
        else:
            n.loads.at[load, "p_set"] = float(n.loads.at[load, "p_set"]) * factor
    return True


def scale_iea_final_loads(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> list[str]:
    """
    Scale fixed demands to the Current Policies final-energy rows.

    Returns the sector-fuel rows that were not found on the network.
    """
    spec = config.get("iea_final") or {}
    if not spec.get("enable") or planning_horizons is None:
        return []
    year = int(planning_horizons)
    nyears = _nyears(n)
    skipped = []
    for (sector, fuel), carriers in LOAD_GROUPS.items():
        pair = (spec.get(sector) or {}).get(fuel)
        if not pair:
            continue
        mtoe = published_value(pair, year)
        loads = _load_index(n, carriers)
        target = mtoe * MWH_PER_MTOE * nyears
        if loads.empty or not _scale_loads(n, loads, target):
            skipped.append(f"{sector} {fuel}")
            logger.warning("No %s %s load to scale to %.1f Mtoe.", sector, fuel, mtoe)
            continue
        logger.info("Scaled %s %s demand to %.1f Mtoe.", sector, fuel, mtoe)
        if (sector, fuel) == ("transport", "oil"):
            _add_biofuels_to_ice(n, spec, year, nyears, skipped)
    return skipped


def _add_biofuels_to_ice(
    n: pypsa.Network, spec: dict, year: int, nyears: float, skipped: list[str]
) -> None:
    """Put transport biofuels on internal-combustion oil."""
    extra = _row_mtoe(spec, "transport", "biofuels", year)
    if extra <= 0:
        return
    ice = _load_index(n, ["land transport oil"])
    added = extra * MWH_PER_MTOE * nyears
    if ice.empty or not _scale_loads(n, ice, _load_energy_mwh(n, ice) + added):
        skipped.append("transport biofuels")
        logger.warning(
            "Transport biofuels of %.1f Mtoe were not added. "
            "No internal-combustion oil load is available.",
            extra,
        )
        return
    logger.info(
        "Transport biofuels of %.1f Mtoe are added to internal-combustion oil. "
        "Biomass-to-liquid is off, and the efficiencies are similar.",
        extra,
    )


def _building_links(n: pypsa.Network, fuel: str) -> pd.Index:
    if n.links.empty:
        return pd.Index([])
    return n.links.index[
        n.links.carrier.map(lambda carrier: _is_buildings_fuel(str(carrier), fuel))
    ]


def _is_buildings_fuel(carrier: str, fuel: str) -> bool:
    name = carrier.lower()
    if fuel == "oil":
        return "oil boiler" in name
    if fuel == "gas":
        return "gas boiler" in name or "micro gas chp" in name
    if fuel == "coal":
        return "coal boiler" in name
    if fuel == "bioenergy":
        return "biomass boiler" in name and "steam" not in name
    return False


def add_iea_final_fuel_constraints(
    n: pypsa.Network, planning_horizons: str | None
) -> None:
    """Fix building fuel input at the Current Policies final-energy rows."""
    spec = (getattr(n, "config", None) or {}).get("iea_final") or {}
    if not spec.get("enable") or planning_horizons is None:
        return
    if "Link-p" not in n.model.variables:
        return
    year = int(planning_horizons)
    weight = n.snapshot_weightings.generators
    nyears = _nyears(n)
    buildings = spec.get("buildings") or {}
    for fuel in ("coal", "oil", "gas", "bioenergy"):
        pair = buildings.get(fuel)
        if not pair:
            continue
        chosen = _building_links(n, fuel)
        mtoe = published_value(pair, year)
        if chosen.empty:
            logger.warning(
                "No building %s plant is available for the final-energy row.", fuel
            )
            continue
        flow = n.model["Link-p"]
        energy = None
        for link_name in chosen:
            term = (flow.loc[:, link_name] * weight).sum()
            energy = term if energy is None else energy + term
        n.model.add_constraints(
            energy == mtoe * MWH_PER_MTOE * nyears,
            name=f"iea_final_buildings_{fuel}",
        )
        logger.info("Building %s use is fixed at %.1f Mtoe.", fuel, mtoe)


def _fossil_fuel(carrier: str) -> str | None:
    """Map a link carrier to coal, oil or gas. ``boiler`` is not oil."""
    name = carrier.lower()
    if "process emission" in name or "biomass" in name or "biogas" in name or "waste" in name:
        return None
    if "lignite" in name or re.search(r"\bcoal\b", name):
        return "coal"
    if re.search(r"\boil\b", name) or "kerosene" in name or "naphtha" in name:
        return "oil"
    if name in {"ocgt", "ccgt", "smr"} or re.search(r"\b(ocgt|ccgt|smr|gas)\b", name):
        return "gas"
    return None


def _atmosphere_efficiency(n: pypsa.Network) -> pd.Series:
    efficiency = pd.Series(0.0, index=n.links.index)
    for column in n.links.columns:
        if not str(column).startswith("bus") or not str(column)[3:].isdigit():
            continue
        port = str(column)[3:]
        buses = n.links[column]
        on_atmosphere = buses.map(n.buses.carrier).eq("co2")
        if port == "0":
            port_efficiency = pd.Series(-1.0, index=n.links.index)
        elif port == "1":
            port_efficiency = n.links["efficiency"]
        elif f"efficiency{port}" in n.links.columns:
            port_efficiency = n.links[f"efficiency{port}"]
        else:
            continue
        efficiency = efficiency + port_efficiency.where(on_atmosphere, 0.0).fillna(0.0)
    return efficiency


def _co2_weights(n: pypsa.Network) -> pd.DataFrame:
    """Split each link's atmosphere CO2 between power, final and other."""
    efficiency = _atmosphere_efficiency(n)
    rows = []
    for name, link in n.links.iterrows():
        fuel = _fossil_fuel(str(link.carrier))
        coeff = float(efficiency.at[name])
        if fuel is None or coeff == 0:
            continue
        carrier = str(link.carrier).lower()
        bus1 = n.buses.carrier.get(link.bus1, "")
        electrical = bus1 in {"AC", "low voltage"}
        chp = "chp" in carrier and electrical
        heat = float(link.efficiency2) if "efficiency2" in n.links.columns else 0.0
        power_share = 0.0
        final_share = 0.0
        if "other energy" in carrier:
            other_share = 1.0
        elif "refining" in carrier or carrier in {"smr", "smr cc"}:
            other_share = 1.0
        elif chp and heat > 0:
            elec = float(link.efficiency)
            power_share = elec / (elec + heat)
            final_share = heat / (elec + heat)
            other_share = 0.0
        elif electrical and "boiler" not in carrier:
            power_share = 1.0
            other_share = 0.0
        else:
            final_share = 1.0
            other_share = 0.0
        transport = any(
            token in carrier
            for token in ("land transport oil", "kerosene", "shipping oil")
        )
        rows.append(
            {
                "link": name,
                "fuel": fuel,
                "coeff": coeff,
                "power": power_share,
                "final": final_share,
                "other": other_share,
                "transport": float(transport),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["link", "fuel", "coeff", "power", "final", "other", "transport"]
        )
    return pd.DataFrame(rows).set_index("link")


def add_iea_combustion_co2_constraints(
    n: pypsa.Network, planning_horizons: str | None
) -> None:
    """
    Fix fossil combustion CO2 at the published Current Policies megatonnes.

    Process emissions are left out. EU oil refining is counted in oil and in
    neither power nor final consumption. Combined-heat-and-power CO2 is split
    by the electrical and heat efficiencies.
    """
    spec = (getattr(n, "config", None) or {}).get("iea_co2") or {}
    if not spec.get("enable") or planning_horizons is None:
        return
    if "Link-p" not in n.model.variables:
        logger.warning("No link flows are available for the combustion CO2 totals.")
        return
    weights = _co2_weights(n)
    if weights.empty:
        logger.warning("No fossil combustion links were found.")
        return
    year = int(planning_horizons)
    snapshot_weight = n.snapshot_weightings.generators
    nyears = _nyears(n)
    groups = {
        "coal": (weights["fuel"].eq("coal"), None),
        "oil": (weights["fuel"].eq("oil"), None),
        "gas": (weights["fuel"].eq("gas"), None),
        "power_coal": (weights["fuel"].eq("coal"), "power"),
        "power_oil": (weights["fuel"].eq("oil"), "power"),
        "power_gas": (weights["fuel"].eq("gas"), "power"),
        "final_coal": (weights["fuel"].eq("coal"), "final"),
        "final_oil": (weights["fuel"].eq("oil"), "final"),
        "final_gas": (weights["fuel"].eq("gas"), "final"),
        "transport_oil": (weights["fuel"].eq("oil") & weights["transport"].eq(1.0), None),
    }
    flow = n.model["Link-p"]
    for name, (mask, column) in groups.items():
        pair = spec.get(name)
        if not pair:
            continue
        chosen = weights.index[mask.to_numpy()]
        if column is not None:
            chosen = chosen[weights.loc[chosen, column] > 0]
        if len(chosen) == 0:
            logger.warning("No links for the %s CO2 row.", name.replace("_", " "))
            continue
        coeff = weights.loc[chosen, "coeff"]
        if column is not None:
            coeff = coeff * weights.loc[chosen, column]
        energy = None
        for link_name in chosen:
            term = (
                flow.loc[:, link_name] * float(coeff.at[link_name]) * snapshot_weight
            ).sum()
            energy = term if energy is None else energy + term
        mt = published_value(pair, year)
        n.model.add_constraints(energy == mt * 1e6 * nyears, name=f"iea_co2_{name}")
        logger.info("Combustion CO2 from %s is fixed at %.0f Mt.", name.replace("_", " "), mt)


def fossil_primary_mwh(n: pypsa.Network) -> dict[str, float]:
    """Fuel supplied by coal, oil and gas generators, in MWh over the snapshots."""
    if n.generators.empty or n.generators_t.p.empty:
        return {}
    carriers = {
        "coal": {"coal", "lignite"},
        "oil": {"oil", "oil primary"},
        "gas": {"gas"},
    }
    weight = n.snapshot_weightings.generators
    totals = {}
    for fuel, names in carriers.items():
        idx = n.generators.index[n.generators.carrier.isin(names)]
        idx = idx.intersection(n.generators_t.p.columns)
        if len(idx) == 0:
            totals[fuel] = 0.0
            continue
        totals[fuel] = float((n.generators_t.p[idx].mul(weight, axis=0)).sum().sum())
    return totals


def log_iea_primary_fuel_check(
    n: pypsa.Network, planning_horizons: str | None
) -> None:
    """
    Compare solved primary fuel use with Table A.1.

    This is a report. It does not change the optimisation. Oil uses 7.33
    barrels per toe. Gas uses 398 Mtoe per 485 bcm. Coal uses 0.7 toe per
    tonne of coal equivalent.
    """
    spec = (getattr(n, "config", None) or {}).get("iea_primary_check") or {}
    if not spec.get("enable") or planning_horizons is None:
        return
    actual = fossil_primary_mwh(n)
    if not actual:
        return
    year = int(planning_horizons)
    targets = {
        "oil": published_value(spec["oil_mbd"], year) * MBD_TO_MWH * _nyears(n),
        "gas": published_value(spec["gas_bcm"], year) * BCM_TO_MWH * _nyears(n),
        "coal": published_value(spec["coal_mtce"], year) * MTCE_TO_MWH * _nyears(n),
    }
    for fuel, target in targets.items():
        used = actual.get(fuel, 0.0)
        logger.info(
            "Primary %s check: model %.1f TWh, Current Policies %.1f TWh.",
            fuel,
            used / 1e6,
            target / 1e6,
        )


def lift_geothermal_if_energy_exceeds_capacity(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Raise geothermal capacity when the published energy exceeds the gigawatts.

    One gigawatt running every hour produces 8.76 TWh. The 2030 row is 1 GW
    and 9 TWh, so the fleet is scaled up, in proportion to the plants already
    there, until full-load operation can meet the energy. The capacity limit
    is moved to that same fleet. Later years are left on the published gigawatts
    when those gigawatts can produce the energy.
    """
    if planning_horizons is None or n.generators.empty:
        return
    spec = config.get("iea_generation") or {}
    pair = spec.get("geothermal") if spec.get("enable") else None
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    if not pair or not options.get("file"):
        return
    year = int(planning_horizons)
    twh = published_value(pair, year)
    from scripts.solve_network import _ccl_horizon_block

    table = pd.read_csv(options["file"], index_col=[0, 1], header=[0, 1])
    block = _ccl_horizon_block(table, year)
    if "geothermal" not in set(block.index.get_level_values("carrier")):
        return
    published_mw = float(block["max"].xs("geothermal", level="carrier").sum())
    required_mw = twh / 8.76 * 1000.0
    if required_mw <= published_mw * (1 + 1e-6):
        return
    idx = n.generators.index[n.generators.carrier.eq("geothermal")]
    if idx.empty:
        logger.warning(
            "Geothermal generation of %.1f TWh needs %.3f GW, and no geothermal "
            "plant is available to raise.",
            twh,
            required_mw / 1000.0,
        )
        return
    weights = n.generators.loc[idx, "p_nom"].fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=idx)
    lifted = weights / float(weights.sum()) * required_mw
    original = n.generators.loc[idx, "p_nom"].fillna(0.0)
    n.generators.loc[idx, "p_nom"] = original.clip(upper=lifted)
    n.generators.loc[idx, "p_nom_min"] = lifted
    n.generators.loc[idx, "p_nom_max"] = lifted
    n.generators.loc[idx, "p_nom_extendable"] = True
    n.generators.loc[idx, "p_max_pu"] = 1.0
    n.generators.loc[idx, "p_min_pu"] = 1.0
    for attr in ("p_max_pu", "p_min_pu"):
        profile = getattr(n.generators_t, attr, None)
        if profile is None or profile.empty:
            continue
        drop = idx.intersection(profile.columns)
        if len(drop):
            profile.drop(columns=list(drop), inplace=True)
    country = n.generators.loc[idx, "bus"].map(n.buses.country)
    override = dict(getattr(n, "_iea_capacity_override", {}))
    for code, group in lifted.groupby(country):
        if pd.isna(code):
            continue
        override[(code, "geothermal")] = float(group.sum())
    n._iea_capacity_override = override
    logger.info(
        "Geothermal capacity is raised from %.3f GW to %.3f GW so %.1f TWh "
        "can be generated at full load.",
        published_mw / 1000.0,
        required_mw / 1000.0,
        twh,
    )


def _supply_bus_and_intensity(n: pypsa.Network, fuel: str) -> tuple[str | None, float | None]:
    """Fuel bus and combustion factor, taken from the existing plant of that fuel."""
    if n.links.empty:
        return None, None
    carriers = n.links.carrier.astype(str)
    if fuel == "oil" and "carrier" in n.buses.columns:
        primary = n.buses.index[n.buses.carrier.astype(str).eq("oil primary")]
        oil_plant = n.links.index[carriers.eq("oil")]
        intensity = None
        if len(oil_plant) and "efficiency2" in n.links.columns:
            value = n.links.at[oil_plant[0], "efficiency2"]
            if pd.notna(value) and float(value) > 0:
                intensity = float(value)
        if len(primary):
            return str(primary[0]), intensity
    plant_carrier = {"coal": "coal", "oil": "oil", "gas": "OCGT"}[fuel]
    chosen = n.links.index[carriers.eq(plant_carrier)]
    if chosen.empty or "efficiency2" not in n.links.columns:
        return None, None
    value = n.links.at[chosen[0], "efficiency2"]
    if pd.isna(value) or float(value) <= 0:
        return None, None
    return str(n.links.at[chosen[0], "bus0"]), float(value)


def add_coal_boilers(n: pypsa.Network, config: dict) -> None:
    """
    Add a coal boiler beside each oil boiler.

    The heat bus, efficiency and capital cost are the oil boiler's. The fuel
    and the carbon dioxide factor are coal's. Building coal then has a plant
    of its own.
    """
    spec = config.get("iea_final") or {}
    if not spec.get("enable") or not (spec.get("buildings") or {}).get("coal"):
        return
    if n.links.empty:
        return
    oil = n.links.index[n.links.carrier.astype(str).str.contains("oil boiler")]
    if oil.empty:
        logger.warning("No oil boiler is available to copy as a coal boiler.")
        return
    bus, intensity = _supply_bus_and_intensity(n, "coal")
    if bus is None or intensity is None:
        logger.warning("Coal boilers need a coal plant to copy the fuel bus and emissions.")
        return
    added = 0
    for name in oil:
        new = str(name).replace("oil boiler", "coal boiler")
        if new in n.links.index:
            continue
        row = n.links.loc[name]
        lifetime = row.lifetime if "lifetime" in n.links.columns else 25.0
        capital = row.capital_cost if "capital_cost" in n.links.columns else 0.0
        n.add(
            "Link",
            new,
            bus0=bus,
            bus1=row.bus1,
            bus2="co2 atmosphere",
            carrier=str(row.carrier).replace("oil boiler", "coal boiler"),
            efficiency=float(row.efficiency),
            efficiency2=intensity,
            capital_cost=0.0 if pd.isna(capital) else float(capital),
            p_nom_extendable=True,
            lifetime=25.0 if pd.isna(lifetime) else float(lifetime),
        )
        added += 1
    if added:
        logger.info("Added %d coal boilers beside the oil boilers.", added)


def add_other_energy_sector(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add one EU link per fuel for combustion outside power and final use.

    The link burns that fuel and sends the carbon dioxide to the atmosphere.
    Power and final rows stay on their own plants, so the fuel total can
    still hold.
    """
    spec = config.get("iea_co2") or {}
    if not spec.get("enable") or planning_horizons is None:
        return
    if "co2 atmosphere" not in n.buses.index:
        logger.warning("No atmosphere bus is available for the other energy sector.")
        return
    year = int(planning_horizons)
    rows = {
        "coal": ("coal", "power_coal", "final_coal"),
        "oil": ("oil", "power_oil", "final_oil"),
        "gas": ("gas", "power_gas", "final_gas"),
    }
    for fuel, keys in rows.items():
        if not all(spec.get(key) for key in keys):
            continue
        name = f"EU other energy {fuel}"
        if name in n.links.index:
            continue
        bus, intensity = _supply_bus_and_intensity(n, fuel)
        if bus is None or intensity is None:
            logger.warning(
                "Other energy sector %s was not added. No %s plant supplies a fuel bus.",
                fuel,
                fuel,
            )
            continue
        n.add(
            "Link",
            name,
            bus0=bus,
            bus1="co2 atmosphere",
            carrier=f"other energy {fuel}",
            p_nom_extendable=True,
            efficiency=intensity,
            capital_cost=0.0,
            marginal_cost=0.0,
        )
        leftover = (
            published_value(spec[keys[0]], year)
            - published_value(spec[keys[1]], year)
            - published_value(spec[keys[2]], year)
        )
        logger.info(
            "Other energy sector %s can emit the %.0f Mt left after power and final use.",
            fuel,
            leftover,
        )


def prepare_iea_cps(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """Derate capacity, add must-run plants, and scale fixed final demands."""
    derate_existing_to_cps(n, config, planning_horizons)
    add_csp_and_marine(n, config, planning_horizons)
    add_coal_boilers(n, config)
    add_other_energy_sector(n, config, planning_horizons)
    scale_iea_final_loads(n, config, planning_horizons)
    lift_geothermal_if_energy_exceeds_capacity(n, config, planning_horizons)
