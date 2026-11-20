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

# Capacity shares are electrical megawatts. Fuel buses are not.
_ELECTRICITY_CARRIERS = {"AC", "low voltage"}

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


def _electrical_capacity_mask(
    n: pypsa.Network, assets: pd.DataFrame, bus_column: str
) -> pd.Series:
    """Assets whose output bus is electricity.

    Biogas, solid biomass and gas supply generators keep an annual energy
    potential in ``p_nom``. That number is not a power-plant rating, so the
    Current Policies electrical share must not scale it.
    """
    return assets[bus_column].map(n.buses.carrier).isin(_ELECTRICITY_CARRIERS)


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
        assets = assets.loc[_electrical_capacity_mask(n, assets, bus_column)]
        if assets.empty:
            continue
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


def _replace_co2_target(
    targets: dict[str, float], name: str, value: float, reason: str
) -> None:
    published = targets[name]
    if abs(published - value) <= 0.05:
        return
    targets[name] = value
    logger.info(
        "Combustion CO2 from %s is set to %.1f Mt. The published figure is %.0f Mt, "
        "and %s.",
        name.replace("_", " "),
        value,
        published,
        reason,
    )


def reconcile_co2_with_fuel_use(n: pypsa.Network, year: int) -> dict[str, float]:
    """
    Move a CO2 total onto the emissions the fixed fuel use can actually produce.

    Oil power plants share one factor, so 21 TWh cannot emit 19 Mt. Transport
    oil, including biofuels burned as oil, emits more than the transport row.
    Final coal has no plant beyond industry and the coal boilers. The fuel
    total keeps the published other-energy residual on top of those rows.
    """
    config = getattr(n, "config", None) or {}
    spec = config.get("iea_co2") or {}
    names = (
        "coal",
        "oil",
        "gas",
        "power_coal",
        "power_oil",
        "power_gas",
        "final_coal",
        "final_oil",
        "final_gas",
        "transport_oil",
    )
    published = {name: published_value(spec[name], year) for name in names}
    targets = dict(published)
    final = config.get("iea_final") or {}
    generation = config.get("iea_generation") or {}
    weights = _co2_weights(n)
    if weights.empty:
        return targets

    def row(sector: str, fuel: str) -> float:
        pair = (final.get(sector) or {}).get(fuel)
        if not pair:
            return 0.0
        early = float(pair.get(2030, pair.get("2030", 0)) or 0)
        late = float(pair.get(2040, pair.get("2040", 0)) or 0)
        if early == 0 and late == 0:
            return 0.0
        return published_value(pair, year)

    def tonnes_per_mwh(carrier_text: str) -> float:
        carriers = n.links.carrier.reindex(weights.index).astype(str)
        chosen = weights.index[carriers.str.contains(carrier_text, case=False, regex=False)]
        if len(chosen) == 0:
            return 0.0
        return float(weights.loc[chosen, "coeff"].median())

    def mt_from_mtoe(mtoe: float, coeff: float) -> float:
        return mtoe * MWH_PER_MTOE * coeff / 1e6

    oil_burn = tonnes_per_mwh("land transport oil")
    transport_mt = mt_from_mtoe(row("transport", "oil") + row("transport", "biofuels"), oil_burn)
    _replace_co2_target(
        targets,
        "transport_oil",
        transport_mt,
        "the transport fuel row burns at the model's oil factor",
    )

    power = weights[weights["fuel"].eq("oil") & weights["power"].gt(0)]
    oil_twh = generation.get("oil")
    if len(power) and oil_twh:
        electrical = n.links.efficiency.reindex(power.index).astype(float).replace(0, pd.NA)
        intensity = float((power["coeff"] * power["power"] / electrical).median())
        _replace_co2_target(
            targets,
            "power_oil",
            published_value(oil_twh, year) * intensity,
            "every oil plant emits the same amount per megawatt-hour",
        )

    final_oil_floor = (
        transport_mt
        + mt_from_mtoe(row("buildings", "oil"), tonnes_per_mwh("oil boiler"))
        + mt_from_mtoe(row("industry", "oil"), tonnes_per_mwh("naphtha"))
    )
    if targets["final_oil"] < final_oil_floor - 0.05:
        _replace_co2_target(
            targets,
            "final_oil",
            final_oil_floor,
            "industry, buildings and transport oil already emit more than that",
        )

    coal_carriers = set(
        n.links.carrier.reindex(
            weights.index[weights["fuel"].eq("coal") & weights["final"].gt(0)]
        ).astype(str)
    )
    boiler_coeff = tonnes_per_mwh("coal boiler") or tonnes_per_mwh("coal for industry")
    final_coal_locked = mt_from_mtoe(
        row("industry", "coal"), tonnes_per_mwh("coal for industry")
    ) + mt_from_mtoe(row("buildings", "coal"), boiler_coeff)
    free_coal = [
        carrier
        for carrier in coal_carriers
        if "boiler" not in carrier.lower() and "industry" not in carrier.lower()
    ]
    if not free_coal and targets["final_coal"] > final_coal_locked + 0.05:
        _replace_co2_target(
            targets,
            "final_coal",
            final_coal_locked,
            "industry and coal boilers are the only final coal plants",
        )

    gas_floor = mt_from_mtoe(
        row("industry", "gas"), tonnes_per_mwh("gas for industry")
    ) + mt_from_mtoe(row("buildings", "gas"), tonnes_per_mwh("gas boiler"))
    if targets["final_gas"] < gas_floor - 0.05:
        _replace_co2_target(
            targets,
            "final_gas",
            gas_floor,
            "industry and gas boilers already emit more than that",
        )

    for fuel in ("coal", "oil", "gas"):
        other = published[fuel] - published[f"power_{fuel}"] - published[f"final_{fuel}"]
        _replace_co2_target(
            targets,
            fuel,
            targets[f"power_{fuel}"] + targets[f"final_{fuel}"] + other,
            "power, final and the other-energy residual have to add up",
        )
    return targets


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
    reconciled = reconcile_co2_with_fuel_use(n, year)
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
        mt = reconciled.get(name, published_value(pair, year))
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
    there, until full-load operation is a little above the energy. The annual
    equality is what forces that energy, so the plants are not held at full
    load: minimum energy stays below the target and maximum stays above it.
    The capacity limit is moved to that same fleet. A later year is left
    alone only when the plants themselves can produce the energy. The
    published gigawatts are not enough of a check: the file can allow more
    capacity than the fleet that was actually built.
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
    idx = n.generators.index[n.generators.carrier.eq("geothermal")]
    if idx.empty:
        logger.warning(
            "Geothermal generation of %.1f TWh needs %.3f GW, and no geothermal "
            "plant is available to raise.",
            twh,
            required_mw / 1000.0,
        )
        return
    installed = pd.to_numeric(n.generators.loc[idx, "p_nom"], errors="coerce")
    if "p_nom_max" in n.generators.columns:
        finite = pd.to_numeric(n.generators.loc[idx, "p_nom_max"], errors="coerce")
        finite = finite.mask(finite.abs() == float("inf"))
        installed = installed.where(finite.isna(), finite)
    if float(installed.sum()) + 1e-6 >= required_mw:
        return
    weights = n.generators.loc[idx, "p_nom"].fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=idx)
    # Full load sits a little above the published terawatt-hours. The annual
    # equality is the target, and the plants have to be able to run below it.
    lifted = weights / float(weights.sum()) * required_mw * (1 + 1e-6)
    n.generators.loc[idx, "p_nom"] = lifted
    n.generators.loc[idx, "p_nom_min"] = lifted
    n.generators.loc[idx, "p_nom_max"] = lifted
    n.generators.loc[idx, "p_nom_extendable"] = True
    n.generators.loc[idx, "p_max_pu"] = 1.0
    n.generators.loc[idx, "p_min_pu"] = 0.0
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
        "Geothermal capacity is raised from %.3f GW to %.3f GW so full load "
        "is above %.1f TWh and the plants can run below that total. "
        "The published cap is %.3f GW.",
        float(installed.sum()) / 1000.0,
        float(lifted.sum()) / 1000.0,
        twh,
        published_mw / 1000.0,
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


def _energy_per_mw(n: pypsa.Network, names: pd.Index) -> float:
    """Megawatt-hours a plant can produce per megawatt, weighted across plants."""
    if len(names) == 0:
        return 0.0
    weight = n.snapshot_weightings.generators
    plants = n.generators.loc[names]
    sizes = plants["p_nom"].fillna(0.0).clip(lower=0.0)
    if float(sizes.sum()) <= 0:
        sizes = pd.Series(1.0, index=plants.index)
    profile = n.generators_t.p_max_pu
    total = 0.0
    for name, size in sizes.items():
        if name in profile.columns:
            per_mw = float((profile[name] * weight).sum())
        else:
            per_mw = float(plants.at[name, "p_max_pu"]) * float(weight.sum())
        total += per_mw * float(size)
    return total / float(sizes.sum())


def _capacity_limits(n: pypsa.Network, config: dict, year: int) -> pd.Series | None:
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    if not options.get("file"):
        return None
    from scripts.solve_network import _ccl_horizon_block

    table = pd.read_csv(options["file"], index_col=[0, 1], header=[0, 1])
    block = _ccl_horizon_block(table, year)
    caps = block["max"].dropna()
    override = getattr(n, "_iea_capacity_override", None) or {}
    for key, value in override.items():
        caps.loc[key] = value[1] if isinstance(value, tuple) else value
    return caps


def lift_weather_limited_generation(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Raise a capacity limit when the weather year cannot produce the terawatt-hours.

    The published gigawatts stay the minimum. The maximum is scaled, by the
    same factor in every country, until the availability profiles can reach
    the electricity total. Nuclear uses the same rule for plant availability.
    """
    if planning_horizons is None or n.generators.empty:
        return
    spec = config.get("iea_generation") or {}
    if not spec.get("enable"):
        return
    year = int(planning_horizons)
    groups = {
        "wind": {
            "onwind": {"onwind"},
            "offwind-all": {"offwind", "offwind-ac", "offwind-dc", "offwind-float"},
        },
        "solar": {
            "solar-all": {"solar", "solar-hsat", "solar rooftop", "csp"},
        },
        "nuclear": {},
    }
    caps = _capacity_limits(n, config, year)
    if caps is None:
        return
    nyears = _nyears(n)
    override = dict(getattr(n, "_iea_capacity_override", {}))

    for name, aggregates in groups.items():
        pair = spec.get(name)
        if not pair:
            continue
        target = published_value(pair, year) * 1e6 * nyears
        if name == "nuclear":
            plants = n.generators.index[n.generators.carrier.eq("nuclear")]
            if plants.empty:
                continue
            available = 0.0
            for plant in plants:
                available += _energy_per_mw(n, pd.Index([plant])) * float(
                    n.generators.at[plant, "p_nom_max"]
                )
            if available >= target:
                continue
            scale = target / available * (1 + 1e-6)
            n.generators.loc[plants, "p_nom_max"] = (
                n.generators.loc[plants, "p_nom_max"] * scale
            )
            country = n.generators.loc[plants, "bus"].map(n.buses.country)
            for code, block in n.generators.loc[plants].groupby(country):
                if pd.isna(code) or (code, "nuclear") not in caps.index:
                    continue
                published = float(caps.loc[(code, "nuclear")])
                override[(code, "nuclear")] = (
                    published,
                    float(n.generators.loc[block.index, "p_nom_max"].sum()),
                )
            logger.info(
                "Nuclear capacity is raised by %.2f%% so plant availability can "
                "generate %.1f TWh.",
                (scale - 1) * 100,
                target / 1e6 / nyears,
            )
            continue

        available = 0.0
        country_caps: list[tuple[str, str, float]] = []
        for aggregate, carriers in aggregates.items():
            plants = n.generators.index[n.generators.carrier.isin(carriers)]
            if plants.empty or aggregate not in set(caps.index.get_level_values("carrier")):
                continue
            chosen = n.generators.loc[plants]
            for code, block in chosen.groupby(chosen.bus.map(n.buses.country)):
                key = (code, aggregate)
                if pd.isna(code) or key not in caps.index:
                    continue
                cap = float(caps.loc[key])
                available += _energy_per_mw(n, block.index) * cap
                country_caps.append((str(code), aggregate, cap))
        if not country_caps or available >= target:
            continue
        scale = target / available * (1 + 1e-6)
        for code, aggregate, cap in country_caps:
            override[(code, aggregate)] = (cap, cap * scale)
            plants = n.generators.index[
                n.generators.carrier.isin(aggregates[aggregate])
            ]
            country = n.generators.loc[plants, "bus"].map(n.buses.country)
            local = plants[country.eq(code).to_numpy()]
            extendable_flag = (
                n.generators.loc[local, "p_nom_extendable"].fillna(False).astype(bool)
            )
            extendable = local[extendable_flag.to_numpy()]
            if len(extendable) == 0:
                continue
            fixed = float(
                n.generators.loc[local.difference(extendable), "p_nom"].fillna(0).sum()
            )
            room = max(0.0, cap * scale - fixed)
            have = float(
                n.generators.loc[extendable, "p_nom_max"].clip(upper=1e12).sum()
            )
            if have >= room:
                continue
            weights = n.generators.loc[extendable, "p_nom"].fillna(0).clip(lower=0)
            if float(weights.sum()) <= 0:
                weights = pd.Series(1.0, index=extendable)
            n.generators.loc[extendable, "p_nom_max"] = (
                weights / float(weights.sum()) * room
            )
        logger.info(
            "%s capacity is raised by %.1f%% so the weather year can generate "
            "%.1f TWh. At the published gigawatts the profiles reach %.1f TWh.",
            name.capitalize(),
            (scale - 1) * 100,
            target / 1e6 / nyears,
            available / 1e6 / nyears,
        )
    n._iea_capacity_override = override


def lift_hydro_inflow_to_published_energy(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add reservoir water when the weather year is below the hydro total.

    Run-of-river is already at its availability in some hours, so it is not
    scaled. The missing energy is added to reservoir inflow, in proportion to
    the water already there. Pumped storage is not hydro generation.
    """
    if planning_horizons is None:
        return
    spec = config.get("iea_generation") or {}
    pair = spec.get("hydro") if spec.get("enable") else None
    if not pair:
        return
    year = int(planning_horizons)
    nyears = _nyears(n)
    target = published_value(pair, year) * 1e6 * nyears
    weight = n.snapshot_weightings.generators
    ror = n.generators.index[n.generators.carrier.eq("ror")] if not n.generators.empty else []
    ror_energy = _energy_per_mw(n, pd.Index(ror)) * float(
        n.generators.loc[ror, "p_nom"].fillna(0).sum()
    ) if len(ror) else 0.0
    if n.storage_units.empty or "hydro" not in set(n.storage_units.carrier):
        inflow_energy = 0.0
        hydro = pd.Index([])
    else:
        hydro = n.storage_units.index[n.storage_units.carrier.eq("hydro")]
        inflow = n.storage_units_t.inflow.reindex(columns=hydro).fillna(0.0)
        inflow_energy = float((inflow.mul(weight, axis=0)).sum().sum())
    available = ror_energy + inflow_energy
    if available >= target or inflow_energy <= 0:
        if available < target:
            logger.warning(
                "Hydro generation of %.1f TWh needs %.1f TWh more water, and "
                "there is no reservoir inflow to raise.",
                target / 1e6 / nyears,
                (target - available) / 1e6 / nyears,
            )
        return
    scale = (target - ror_energy) / inflow_energy * (1 + 1e-6)
    n.storage_units_t.inflow.loc[:, hydro] = (
        n.storage_units_t.inflow.loc[:, hydro].fillna(0.0) * scale
    )
    logger.info(
        "Reservoir hydro inflow is raised by %.1f%%. Run-of-river can generate "
        "%.1f TWh and the weather year has %.1f TWh of water, against %.1f TWh.",
        (scale - 1) * 100,
        ror_energy / 1e6 / nyears,
        available / 1e6 / nyears,
        target / 1e6 / nyears,
    )


def remove_agriculture_oil_outside_the_oil_rows(n: pypsa.Network) -> None:
    """Drop agriculture oil that sits on top of the allocated oil rows.

    Transport, buildings and industry already take the whole final-oil row.
    The extra machinery load cannot burn inside that carbon total, so the
    solved 2030 network shed it, about 100 TWh, at the load-shedding price.
    """
    loads = _load_index(n, ["agriculture machinery oil"])
    if len(loads) == 0:
        return
    removed = _load_energy_mwh(n, loads)
    if removed <= 1.0:
        return
    timed = n.loads_t.p_set
    for load in loads:
        n.loads.at[load, "p_set"] = 0.0
        if load in timed.columns:
            timed[load] = 0.0
    logger.info(
        "Agriculture machinery oil of %.1f TWh is removed. Transport, buildings "
        "and industry already use the whole oil row, and the solved network "
        "was shedding this extra demand.",
        removed / 1e6,
    )


def _positive_efficiencies(n: pypsa.Network, carriers: set[str]) -> pd.Series:
    if n.links.empty or "efficiency" not in n.links.columns:
        return pd.Series(dtype=float)
    chosen = n.links.index[n.links.carrier.isin(carriers)]
    if len(chosen) == 0:
        return pd.Series(dtype=float)
    efficiency = pd.to_numeric(n.links.loc[chosen, "efficiency"], errors="coerce")
    return efficiency[efficiency > 0]


def raise_solid_biomass_ceiling(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """Raise the solid-biomass cap when the imposed rows need more fuel.

    The 2030 solution used the whole potential, 1,099 TWh, and shed another
    943 TWh. Biomass electricity at the CHP efficiency, building bioenergy
    and industry bioenergy together ask for about twice that. The ceiling is
    an upper bound, so the extra potential is used only as far as those rows
    require. The unsustainable equality is left as published by the potential.
    """
    if planning_horizons is None:
        return
    year = int(planning_horizons)
    nyears = _nyears(n)
    final = config.get("iea_final") or {}
    generation = config.get("iea_generation") or {}
    buildings = 0.0
    if final.get("enable"):
        buildings = _row_mtoe(final, "buildings", "bioenergy", year) * MWH_PER_MTOE * nyears
    industry_loads = _load_index(n, ["solid biomass for industry"])
    industry = _load_energy_mwh(n, industry_loads) if len(industry_loads) else 0.0
    industry_eff = _positive_efficiencies(n, {"solid biomass for industry"})
    if industry > 0 and len(industry_eff):
        industry = industry / float(industry_eff.min())
    elif industry <= 0 and final.get("enable"):
        industry = _row_mtoe(final, "industry", "bioenergy", year) * MWH_PER_MTOE * nyears
    electricity = 0.0
    if generation.get("enable") and generation.get("biomass"):
        electricity = published_value(generation["biomass"], year) * 1e6 * nyears
    chp_eff = _positive_efficiencies(
        n,
        {
            "urban central solid biomass CHP",
            "urban central solid biomass CHP CC",
            "biomass",
        },
    )
    fuel_for_electricity = 0.0
    if electricity > 0 and len(chp_eff):
        # The least efficient plant has to be able to meet the terawatt-hours.
        fuel_for_electricity = electricity / float(chp_eff.min())
    elif electricity > 0:
        logger.warning(
            "Biomass electricity of %.1f TWh has no solid-biomass CHP efficiency, "
            "so the fuel ceiling is not raised for it.",
            electricity / 1e6,
        )
    required = buildings + industry + fuel_for_electricity
    if required <= 0 or n.generators.empty:
        return
    supply = n.generators.index[n.generators.carrier.eq("solid biomass")]
    if "e_sum_max" not in n.generators.columns or len(supply) == 0:
        return
    finite = n.generators.loc[supply, "e_sum_max"].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    finite = finite.dropna().astype(float)
    generator_cap = float(finite.sum()) if len(finite) else 0.0
    constraint_cap = None
    if "biomass limit" in n.global_constraints.index:
        constraint_cap = float(n.global_constraints.at["biomass limit", "constant"])
    if generator_cap > 0 and constraint_cap is not None:
        sustainable = min(generator_cap, constraint_cap)
    elif generator_cap > 0:
        sustainable = generator_cap
    elif constraint_cap is not None:
        sustainable = constraint_cap
    else:
        return
    unsustainable = 0.0
    if "unsustainable biomass limit" in n.global_constraints.index:
        unsustainable = float(
            n.global_constraints.at["unsustainable biomass limit", "constant"]
        )
    available = sustainable + max(unsustainable, 0.0)
    if required <= available * (1 + 1e-6):
        return
    new_sustainable = required * (1 + 1e-4) - max(unsustainable, 0.0)
    if generator_cap > 0 and len(finite):
        factor = new_sustainable / generator_cap
        n.generators.loc[finite.index, "e_sum_max"] = finite * factor
        same_units = n.generators.loc[finite.index, "p_nom"].astype(float)
        matched = (same_units - finite).abs() <= 1.0
        chosen = matched.index[matched.to_numpy()]
        n.generators.loc[chosen, "p_nom"] = n.generators.loc[chosen, "e_sum_max"]
    if "biomass limit" in n.global_constraints.index:
        n.global_constraints.at["biomass limit", "constant"] = new_sustainable
    logger.info(
        "Solid biomass ceiling is raised from %.1f TWh to %.1f TWh. "
        "Electricity, buildings and industry need %.1f TWh of fuel and the "
        "potentials provide %.1f TWh, of which %.1f TWh is the unsustainable equality.",
        sustainable / 1e6,
        new_sustainable / 1e6,
        required / 1e6,
        available / 1e6,
        max(unsustainable, 0.0) / 1e6,
    )


def _mtoe_share(spec: dict, sector: str, fuel: str, others: list[str], year: int) -> float:
    """Share of a final-energy fuel that belongs to one sector."""
    own = _row_mtoe(spec, sector, fuel, year)
    total = own + sum(_row_mtoe(spec, other, fuel, year) for other in others)
    if total <= 0:
        return 0.0
    return own / total


def industry_co2_target_mt(config: dict, year: int) -> float:
    """
    Industry share of the published final CO2 rows, in megatonnes.

    Current Policies publishes final CO2 by fuel, not a separate industry
    total. Coal and gas are split with buildings by their final-energy rows.
    Oil drops the published transport row first, then splits the remainder
    with buildings.
    """
    final = config.get("iea_final") or {}
    co2 = config.get("iea_co2") or {}
    if not co2:
        return 0.0
    coal = _mtoe_share(final, "industry", "coal", ["buildings"], year)
    gas = _mtoe_share(final, "industry", "gas", ["buildings"], year)
    oil_energy = _row_mtoe(final, "industry", "oil", year) + _row_mtoe(final, "buildings", "oil", year)
    oil_share = 0.0
    if oil_energy > 0:
        oil_share = _row_mtoe(final, "industry", "oil", year) / oil_energy
    total = 0.0
    if "final_coal" in co2:
        total += coal * published_value(co2["final_coal"], year)
    if "final_gas" in co2:
        total += gas * published_value(co2["final_gas"], year)
    if "final_oil" in co2 and "transport_oil" in co2:
        non_transport = published_value(co2["final_oil"], year) - published_value(co2["transport_oil"], year)
        total += oil_share * non_transport
    return total


def _industry_fuel_co2_mt(n: pypsa.Network, config: dict, year: int) -> float:
    """CO2 the fixed industry coal, gas and naphtha rows emit to the atmosphere."""
    spec = config.get("iea_final") or {}
    atmosphere = _atmosphere_efficiency(n)
    carriers = {
        "coal": "coal for industry",
        "gas": "gas for industry",
        "oil": "naphtha for industry",
    }
    total = 0.0
    for fuel, carrier in carriers.items():
        mtoe = _row_mtoe(spec, "industry", fuel, year)
        links = n.links.index[n.links.carrier.astype(str).eq(carrier)] if not n.links.empty else []
        if mtoe <= 0 or len(links) == 0:
            continue
        delivered = n.links.loc[links, "efficiency"].astype(float).replace(0, pd.NA)
        per_mwh = (atmosphere.reindex(links) / delivered).astype(float).dropna()
        if per_mwh.empty:
            continue
        total += mtoe * MWH_PER_MTOE * float(per_mwh.median()) / 1e6
    return total


def _process_co2_mt(n: pypsa.Network) -> float:
    """Process CO2 from the industrial load plus the naphtha steam-cracker slice."""
    loads = (
        n.loads.index[n.loads.carrier.astype(str).eq("process emissions")]
        if not n.loads.empty
        else pd.Index([])
    )
    from_load = -_load_energy_mwh(n, loads) if len(loads) else 0.0
    feedstock = 0.0
    oil = _load_energy_mwh(n, _load_index(n, ["naphtha for industry"]))
    if not n.links.empty and oil:
        links = n.links.index[n.links.carrier.astype(str).eq("naphtha for industry")]
        if len(links) and "efficiency2" in n.links.columns:
            delivered = n.links.loc[links, "efficiency"].astype(float).replace(0, pd.NA)
            ratio = (n.links.loc[links, "efficiency2"].astype(float) / delivered).dropna()
            if not ratio.empty:
                feedstock = oil * float(ratio.median())
    return (from_load + feedstock) / 1e6 / _nyears(n)


def _scale_load_values(n: pypsa.Network, loads: pd.Index, factor: float) -> None:
    timed = n.loads_t.p_set
    for load in loads:
        if load in timed.columns and timed[load].notna().any():
            n.loads_t.p_set[load] = timed[load].fillna(0.0) * factor
        else:
            n.loads.at[load, "p_set"] = float(n.loads.at[load, "p_set"]) * factor


def scale_process_emissions_to_industry_co2(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Scale process CO2 so industry fuel plus process CO2 matches the industry share.

    The fuel rows stay at the published megatonnes of oil equivalent. Process
    emissions, including the steam-cracker slice of naphtha, take the remainder
    of the industry share. Plastic-waste CO2 on the naphtha link stays in the
    oil row.
    """
    if planning_horizons is None or not (config.get("iea_co2") or {}).get("enable"):
        return
    if not (config.get("iea_final") or {}).get("enable"):
        return
    year = int(planning_horizons)
    target = industry_co2_target_mt(config, year)
    fuel = _industry_fuel_co2_mt(n, config, year)
    current = _process_co2_mt(n)
    if current <= 1e-6:
        logger.warning("No process emissions are available to match the industry CO2 share.")
        return
    remainder = target - fuel
    factor = 0.0 if remainder <= 0 else remainder / current
    loads = (
        n.loads.index[n.loads.carrier.astype(str).eq("process emissions")]
        if not n.loads.empty
        else pd.Index([])
    )
    if len(loads):
        _scale_load_values(n, loads, factor)
    links = (
        n.links.index[n.links.carrier.astype(str).eq("naphtha for industry")]
        if not n.links.empty
        else pd.Index([])
    )
    if len(links) and "efficiency2" in n.links.columns:
        n.links.loc[links, "efficiency2"] = n.links.loc[links, "efficiency2"].astype(float) * factor
    logger.info(
        "Process emissions are scaled by %.3f. Industry fuel emits %.1f Mt and "
        "the industry share of the final CO2 rows is %.1f Mt.",
        factor,
        fuel,
        target,
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
    scale_process_emissions_to_industry_co2(n, config, planning_horizons)
    remove_agriculture_oil_outside_the_oil_rows(n)
    raise_solid_biomass_ceiling(n, config, planning_horizons)
    lift_geothermal_if_energy_exceeds_capacity(n, config, planning_horizons)
    lift_weather_limited_generation(n, config, planning_horizons)
    lift_hydro_inflow_to_published_energy(n, config, planning_horizons)
