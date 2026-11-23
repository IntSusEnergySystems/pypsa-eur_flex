# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Compare a solved Current Policies run with the WEO 2019 EU totals.

Reads the solved networks, writes a table, and draws one graph for
electricity, capacity, combustion CO2 and final energy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import yaml
from snakemake.utils import update_config

from scripts.iea_cps_constraints import (
    BCM_TO_MWH,
    LOAD_GROUPS,
    MBD_TO_MWH,
    MTCE_TO_MWH,
    MWH_PER_MTOE,
    _atmosphere_efficiency,
    _building_links,
    _co2_weights,
    _load_energy_mwh,
    _load_index,
    _nyears,
    _row_mtoe,
    fossil_primary_mwh,
    industry_fuel_share_mt,
    published_value,
)
from scripts.solve_network import _ccl_horizon_block, _ccl_rename

logger = logging.getLogger(__name__)

ELECTRICITY = {"AC", "low voltage"}
GENERATION = {
    "coal": ("Link", {"coal", "lignite"}),
    "oil": ("Link", {"oil"}),
    "gas": ("Link", {"OCGT", "CCGT", "gas"}),
    "nuclear": ("Generator", {"nuclear"}),
    "hydro": ("Generator", {"hydro", "ror"}),
    "wind": (
        "Generator",
        {"onwind", "offwind", "offwind-ac", "offwind-dc", "offwind-float"},
    ),
    "solar": ("Generator", {"solar", "solar-hsat", "solar rooftop", "csp"}),
    "geothermal": ("Generator", {"geothermal"}),
    "biomass": (
        "both",
        {
            "biomass",
            "biogas",
            "waste",
            "urban central solid biomass CHP",
            "urban central solid biomass CHP CC",
        },
    ),
}
CO2_GROUPS = {
    "coal": ("coal", None),
    "oil": ("oil", None),
    "gas": ("gas", None),
    "power coal": ("coal", "power"),
    "power oil": ("oil", "power"),
    "power gas": ("gas", "power"),
    "final coal": ("coal", "final"),
    "final oil": ("oil", "final"),
    "final gas": ("gas", "final"),
    "transport oil": ("transport", None),
}
TOLERANCE = {"TWh": 0.1, "GW": 0.02, "Mt": 0.5, "Mtoe": 0.1, "mb/d": 0.05, "bcm": 1.0, "Mtce": 0.5}


def load_config(path: Path | None = None) -> dict:
    """Merge the default configuration with the Current Policies overlay."""
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "config" / "config.default.yaml").read_text())
    overlay = path or (root / "config" / "config.yaml")
    update_config(base, yaml.safe_load(overlay.read_text()) or {})
    return base


def _weight(n: pypsa.Network) -> pd.Series:
    return n.snapshot_weightings.generators


def _built_mw(static: pd.DataFrame) -> pd.Series:
    if "p_nom_opt" in static.columns:
        built = static["p_nom_opt"].where(static["p_nom_opt"].notna(), static["p_nom"])
    else:
        built = static["p_nom"]
    return built.fillna(0.0)


def _annual_mwh(series: pd.Series, weight: pd.Series) -> float:
    return float((series.fillna(0.0) * weight).sum())


def generation_twh(n: pypsa.Network) -> dict[str, float]:
    """EU electricity by the same groups as the generation constraints."""
    weight = _weight(n)
    nyears = _nyears(n)
    totals = {}
    for name, (kind, carriers) in GENERATION.items():
        energy = 0.0
        if kind in {"Generator", "both"} and not n.generators.empty:
            idx = n.generators.index[n.generators.carrier.isin(carriers)]
            bus_carrier = n.generators.loc[idx, "bus"].map(n.buses.carrier)
            idx = bus_carrier.index[bus_carrier.isin(ELECTRICITY).to_numpy()]
            columns = idx.intersection(n.generators_t.p.columns)
            if len(columns):
                energy += float((n.generators_t.p[columns].mul(weight, axis=0)).sum().sum())
        if name == "hydro" and not n.storage_units.empty:
            dispatch = getattr(n.storage_units_t, "p_dispatch", None)
            if dispatch is not None and not dispatch.empty:
                stores = n.storage_units.index[n.storage_units.carrier.eq("hydro")]
                columns = stores.intersection(dispatch.columns)
                if len(columns):
                    energy += float((dispatch[columns].mul(weight, axis=0)).sum().sum())
        if kind in {"Link", "both"} and not n.links.empty:
            chosen = n.links.carrier.isin(carriers)
            bus_carrier = n.links.loc[chosen, "bus1"].map(n.buses.carrier)
            idx = bus_carrier.index[bus_carrier.isin(ELECTRICITY).to_numpy()]
            for unit in idx.intersection(n.links_t.p0.columns):
                energy += _annual_mwh(n.links_t.p0[unit], weight) * float(
                    n.links.at[unit, "efficiency"]
                )
        totals[name] = energy / 1e6 / nyears
    for carrier in ("csp", "marine"):
        if n.generators.empty:
            totals[carrier] = 0.0
            continue
        idx = n.generators.index[n.generators.carrier.eq(carrier)]
        energy = 0.0
        for unit in idx.intersection(n.generators_t.p.columns):
            energy += _annual_mwh(n.generators_t.p[unit], weight)
        totals[carrier] = energy / 1e6 / nyears
    return totals


def capacity_gw(n: pypsa.Network, config: dict, year: int) -> dict[str, float]:
    """EU electrical capacity on the names used in the aggregate limit file."""
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    path = options.get("file")
    if not path or n.buses.empty:
        return {}
    table = pd.read_csv(path, index_col=[0, 1], header=[0, 1])
    block = _ccl_horizon_block(table, year)
    rename = _ccl_rename(block, options)
    if "biomass" in set(block.index.get_level_values("carrier")):
        rename = {
            **rename,
            "urban central solid biomass CHP": "biomass",
            "urban central solid biomass CHP CC": "biomass",
        }
    allowed = set(block.index.get_level_values("carrier"))
    totals: dict[str, float] = {}
    csp_gw = 0.0

    def _add(carrier: str, electrical_mw: float) -> None:
        nonlocal csp_gw
        if carrier == "csp":
            csp_gw += electrical_mw / 1000.0
        name = rename.get(carrier, carrier)
        if name not in allowed and carrier not in {"csp", "marine"}:
            return
        totals[name] = totals.get(name, 0.0) + electrical_mw / 1000.0

    if not n.generators.empty:
        built = _built_mw(n.generators)
        bus_carrier = n.generators.bus.map(n.buses.carrier)
        keep = bus_carrier.isin(ELECTRICITY) | n.generators.carrier.isin(["csp", "marine"])
        for unit in n.generators.index[keep.fillna(False)]:
            _add(str(n.generators.at[unit, "carrier"]), float(built.at[unit]))
    if not n.links.empty:
        built = _built_mw(n.links)
        bus_carrier = n.links.bus1.map(n.buses.carrier)
        for unit in n.links.index[bus_carrier.isin(ELECTRICITY).fillna(False)]:
            efficiency = n.links.at[unit, "efficiency"]
            electrical = float(built.at[unit]) * (0.0 if pd.isna(efficiency) else float(efficiency))
            _add(str(n.links.at[unit, "carrier"]), electrical)
    if not n.storage_units.empty:
        built = _built_mw(n.storage_units)
        hydro = n.storage_units.index[n.storage_units.carrier.eq("hydro")]
        for unit in hydro:
            _add("hydro", float(built.at[unit]))
    totals["wind"] = totals.get("onwind", 0.0) + totals.get("offwind-all", 0.0)
    totals["csp"] = csp_gw
    return totals


def _published_capacity_gw(config: dict, year: int) -> dict[str, float]:
    options = config.get("solving", {}).get("agg_p_nom_limits") or {}
    path = options.get("file")
    if not path:
        return {}
    table = pd.read_csv(path, index_col=[0, 1], header=[0, 1])
    block = _ccl_horizon_block(table, year)
    totals = block["max"].groupby(level="carrier").sum() / 1000.0
    published = {str(name): float(value) for name, value in totals.items()}
    published["wind"] = published.get("onwind", 0.0) + published.get("offwind-all", 0.0)
    must = config.get("iea_must_run") or {}
    if must.get("enable"):
        for carrier in ("csp", "marine"):
            pair = (must.get(carrier) or {}).get("gw")
            if pair:
                published[carrier] = published_value(pair, year)
    return published


def _add_process_mt(
    n: pypsa.Network,
    totals: dict[str, float],
    year: int,
    weight: pd.Series,
    nyears: float,
) -> None:
    """Add solved process CO2 onto the fuel rows in the industry-share split."""
    if n.links.empty or n.links_t.p0.empty or nyears <= 0:
        return
    links = n.links.index[
        n.links.carrier.astype(str).str.contains("process emission", case=False)
    ].intersection(n.links_t.p0.columns)
    if len(links) == 0:
        return
    atmosphere = _atmosphere_efficiency(n)
    tonnes = sum(
        _annual_mwh(n.links_t.p0[unit], weight) * float(atmosphere.at[unit])
        for unit in links
    )
    if tonnes <= 0:
        return
    shares = industry_fuel_share_mt(getattr(n, "config", None) or {}, year)
    industry = sum(shares.values())
    if industry <= 0:
        return
    for fuel, share in shares.items():
        part = tonnes * share / industry / 1e6 / nyears
        totals[fuel] = totals.get(fuel, 0.0) + part
        totals[f"final {fuel}"] = totals.get(f"final {fuel}", 0.0) + part


def co2_mt(n: pypsa.Network, year: int | None = None) -> dict[str, float]:
    """Combustion CO2 by the same split as the optimisation.

    Process emissions are added to each fuel in proportion to the industry
    share, which is how the optimisation counts them.
    """
    weights = _co2_weights(n)
    if weights.empty or n.links_t.p0.empty:
        return {}
    weight = _weight(n)
    nyears = _nyears(n)
    totals = {}
    for label, (fuel, column) in CO2_GROUPS.items():
        if fuel == "transport":
            mask = weights["fuel"].eq("oil") & weights["transport"].eq(1.0)
            column = None
        else:
            mask = weights["fuel"].eq(fuel)
        chosen = weights.index[mask.to_numpy()]
        if column is not None:
            chosen = chosen[weights.loc[chosen, column] > 0]
        tonnes = 0.0
        for unit in chosen.intersection(n.links_t.p0.columns):
            coeff = float(weights.at[unit, "coeff"])
            if column is not None:
                coeff *= float(weights.at[unit, column])
            tonnes += _annual_mwh(n.links_t.p0[unit], weight) * coeff
        totals[label] = tonnes / 1e6 / nyears
    if year is not None:
        _add_process_mt(n, totals, year, weight, nyears)
    for fuel in ("coal", "oil", "gas"):
        chosen = weights.index[weights["fuel"].eq(fuel) & (weights["other"] > 0)]
        tonnes = 0.0
        link_tonnes = 0.0
        for unit in chosen.intersection(n.links_t.p0.columns):
            part = (
                _annual_mwh(n.links_t.p0[unit], weight)
                * float(weights.at[unit, "coeff"])
                * float(weights.at[unit, "other"])
            )
            tonnes += part
            if "other energy" in str(n.links.at[unit, "carrier"]):
                link_tonnes += part
        totals[f"other {fuel}"] = tonnes / 1e6 / nyears
        totals[f"other link {fuel}"] = link_tonnes / 1e6 / nyears
    return totals


def final_mtoe(n: pypsa.Network, config: dict, year: int) -> dict[str, float]:
    """Final energy on the loads and boilers the optimisation scales."""
    spec = config.get("iea_final") or {}
    if not spec.get("enable"):
        return {}
    nyears = _nyears(n)
    weight = _weight(n)
    totals = {}
    for sector, fuel in LOAD_GROUPS:
        loads = _load_index(n, LOAD_GROUPS[(sector, fuel)])
        energy = _load_energy_mwh(n, loads) if len(loads) else 0.0
        totals[f"{sector} {fuel}"] = energy / MWH_PER_MTOE / nyears
    if not n.links.empty:
        for fuel in ("coal", "oil", "gas", "bioenergy"):
            chosen = _building_links(n, fuel).intersection(n.links_t.p0.columns)
            energy = sum(_annual_mwh(n.links_t.p0[unit], weight) for unit in chosen)
            totals[f"buildings {fuel}"] = energy / MWH_PER_MTOE / nyears
    return totals


def _row(category, name, unit, year, model, target, note="", status=""):
    gap = None if target is None else float(model) - float(target)
    if not status:
        tol = TOLERANCE[unit]
        status = "pass" if gap is not None and abs(gap) <= tol else "fail"
    return {
        "category": category,
        "name": name,
        "unit": unit,
        "year": year,
        "model": model,
        "target": target,
        "gap": gap,
        "status": status,
        "note": note,
    }


def compare_network(n: pypsa.Network, config: dict, year: int) -> pd.DataFrame:
    """One row per EU total, with the model value beside the outlook value."""
    rows = []
    spec = config.get("iea_generation") or {}
    actual = generation_twh(n)
    if spec.get("enable"):
        for name in GENERATION:
            if not spec.get(name):
                continue
            rows.append(
                _row("electricity", name, "TWh", year, actual.get(name, 0.0), published_value(spec[name], year))
            )
        must = config.get("iea_must_run") or {}
        if must.get("enable"):
            for carrier in ("csp", "marine"):
                pair = (must.get(carrier) or {}).get("twh")
                if pair:
                    rows.append(
                        _row("electricity", carrier, "TWh", year, actual.get(carrier, 0.0), published_value(pair, year))
                    )
    have = capacity_gw(n, config, year)
    published = _published_capacity_gw(config, year)
    for name, target in published.items():
        note = ""
        if name == "solar-all" and have.get("csp"):
            note = f"Includes {have['csp']:.1f} GW of concentrating solar."
        if name == "geothermal" and abs(have.get(name, 0.0) - float(target)) > 0.02:
            note = "Sized to the terawatt-hours. The bar is the published gigawatts."
        if name == "hydro":
            note = "Run-of-river plus reservoirs. Pumped storage is excluded."
        rows.append(_row("capacity", name, "GW", year, have.get(name, 0.0), target, note))
    co2_spec = config.get("iea_co2") or {}
    emitted = co2_mt(n, year)
    if co2_spec.get("enable"):
        for label, key in (
            ("coal", "coal"),
            ("oil", "oil"),
            ("gas", "gas"),
            ("power coal", "power_coal"),
            ("power oil", "power_oil"),
            ("power gas", "power_gas"),
            ("final coal", "final_coal"),
            ("final oil", "final_oil"),
            ("final gas", "final_gas"),
            ("transport oil", "transport_oil"),
        ):
            if co2_spec.get(key):
                rows.append(
                    _row("co2", label, "Mt", year, emitted.get(label, 0.0), published_value(co2_spec[key], year))
                )
        if all(co2_spec.get(key) for key in ("coal", "oil", "gas")):
            model_total = sum(emitted.get(fuel, 0.0) for fuel in ("coal", "oil", "gas"))
            published_total = sum(published_value(co2_spec[key], year) for key in ("coal", "oil", "gas"))
            rows.append(
                _row(
                    "co2",
                    "total",
                    "Mt",
                    year,
                    model_total,
                    published_total,
                    "Sum of coal, oil and gas combustion. Process emissions are outside this total.",
                )
            )
        for fuel, total_key, power_key, final_key in (
            ("coal", "coal", "power_coal", "final_coal"),
            ("oil", "oil", "power_oil", "final_oil"),
            ("gas", "gas", "power_gas", "final_gas"),
        ):
            if not all(co2_spec.get(key) for key in (total_key, power_key, final_key)):
                continue
            residual = (
                published_value(co2_spec[total_key], year)
                - published_value(co2_spec[power_key], year)
                - published_value(co2_spec[final_key], year)
            )
            rows.append(_row("co2", f"other {fuel}", "Mt", year, emitted.get(f"other {fuel}", 0.0), residual))
            rows.append(
                _row(
                    "other energy",
                    fuel,
                    "Mt",
                    year,
                    emitted.get(f"other link {fuel}", 0.0),
                    None,
                    "Emissions of the added other-energy link. Refining and steam reforming stay in the other total.",
                    "check",
                )
            )
    final_spec = config.get("iea_final") or {}
    used = final_mtoe(n, config, year)
    if final_spec.get("enable"):
        for sector, fuel in list(LOAD_GROUPS) + [("buildings", fuel) for fuel in ("coal", "oil", "gas", "bioenergy")]:
            key = f"{sector} {fuel}"
            if key in {f"{a} {b}" for a, b in LOAD_GROUPS} and (sector, fuel) in {
                ("buildings", "coal"),
                ("buildings", "oil"),
                ("buildings", "gas"),
                ("buildings", "bioenergy"),
            }:
                continue
            target = _row_mtoe(final_spec, sector, fuel, year)
            note = ""
            if (sector, fuel) == ("transport", "oil"):
                extra = _row_mtoe(final_spec, "transport", "biofuels", year)
                note = (
                    f"Original oil row, without {extra:.1f} Mtoe of biofuels. "
                    "Those biofuels are burned inside this oil load."
                )
            if (sector, fuel) == ("industry", "oil"):
                note = "This is the naphtha load."
            if target or key in used:
                rows.append(_row("final energy", key, "Mtoe", year, used.get(key, 0.0), target, note))
    primary = config.get("iea_primary_check") or {}
    if primary.get("enable"):
        raw = fossil_primary_mwh(n)
        nyears = _nyears(n)
        converted = {
            "oil": (raw.get("oil", 0.0) / nyears / MBD_TO_MWH, "oil_mbd", "mb/d"),
            "gas": (raw.get("gas", 0.0) / nyears / BCM_TO_MWH, "gas_bcm", "bcm"),
            "coal": (raw.get("coal", 0.0) / nyears / MTCE_TO_MWH, "coal_mtce", "Mtce"),
        }
        for name, (model, key, unit) in converted.items():
            rows.append(
                _row(
                    "primary",
                    name,
                    unit,
                    year,
                    model,
                    published_value(primary[key], year),
                    "Reported after the solve. Not a constraint.",
                    "check",
                )
            )
    if (config.get("iea_final") or {}).get("buildings", {}).get("coal"):
        boilers = (
            0
            if n.links.empty
            else int(n.links.carrier.astype(str).str.contains("coal boiler").sum())
        )
        rows.append(
            {
                "category": "technology",
                "name": "coal boilers",
                "unit": "links",
                "year": year,
                "model": boilers,
                "target": None,
                "gap": None,
                "status": "pass" if boilers else "fail",
                "note": "Coal boilers are present for the building-coal row.",
            }
        )
    rows.extend(_cost_rows(n, year))
    return pd.DataFrame(rows)


def _as_carrier_series(value) -> pd.Series:
    """Collapse a statistics result to one number per carrier."""
    if isinstance(value, pd.DataFrame):
        value = value.sum(axis=1)
    if not isinstance(value, pd.Series) or value.empty:
        return pd.Series(dtype=float)
    if isinstance(value.index, pd.MultiIndex):
        value = value.groupby(value.index.get_level_values(-1)).sum()
    value.index = value.index.map(str)
    return value.groupby(level=0).sum()


def _cost_rows(n: pypsa.Network, year: int) -> list[dict]:
    """Model investment and operation. Current Policies has no euro total here."""
    try:
        new_investment = _as_carrier_series(n.statistics.expanded_capex())
        existing = _as_carrier_series(n.statistics.installed_capex())
        operation = _as_carrier_series(n.statistics.opex())
    except Exception:
        logger.warning("System cost could not be read for %s.", year)
        return []
    note = (
        "Billion euro per year. Operation and new investment are in the objective. "
        "Existing capital is capacity already built. "
        "Current Policies does not publish this total in euro."
    )
    rows = [
        _row("cost", "new investment", "bn EUR", year, float(new_investment.sum()) / 1e9, None, note, "check"),
        _row("cost", "operation", "bn EUR", year, float(operation.sum()) / 1e9, None, note, "check"),
        _row("cost", "existing capital", "bn EUR", year, float(existing.sum()) / 1e9, None, note, "check"),
    ]
    objective = getattr(n, "objective", None)
    if objective is not None and pd.notna(objective):
        rows.append(
            _row(
                "cost",
                "total",
                "bn EUR",
                year,
                float(objective) / 1e9,
                None,
                note,
                "check",
            )
        )
    combined = new_investment.add(operation, fill_value=0.0)
    combined = combined[combined.abs() > 1e6].sort_values(ascending=False)
    for carrier, value in combined.items():
        rows.append(
            _row(
                "cost carrier",
                str(carrier),
                "bn EUR",
                year,
                float(value) / 1e9,
                None,
                "Operation plus new investment for this carrier.",
                "check",
            )
        )
    return rows


MODEL_COLOR = "#1f4e79"
CPS_COLOR = "#c47b2b"
ORDER = {
    "electricity": [
        "coal", "oil", "gas", "nuclear", "hydro", "wind", "solar",
        "geothermal", "biomass", "csp", "marine",
    ],
    "capacity": [
        "coal-all", "oil", "gas", "nuclear", "hydro", "biomass",
        "onwind", "offwind-all", "solar-all", "geothermal", "marine",
    ],
    "co2": [
        "total", "coal", "oil", "gas",
        "power coal", "power oil", "power gas",
        "final coal", "final oil", "final gas",
        "transport oil", "other coal", "other oil", "other gas",
    ],
    "final energy": [
        "industry coal", "industry oil", "industry gas", "industry electricity",
        "industry heat", "industry bioenergy",
        "transport oil", "transport electricity", "transport biofuels",
        "buildings coal", "buildings oil", "buildings gas", "buildings electricity",
        "buildings heat", "buildings bioenergy",
    ],
}


def _ordered_names(part: pd.DataFrame, category: str) -> list[str]:
    present = list(dict.fromkeys(part.name))
    preferred = [name for name in ORDER.get(category, []) if name in present]
    rest = [name for name in present if name not in preferred]
    return preferred + rest


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_category(frame: pd.DataFrame, category: str, path: Path) -> None:
    """One panel per year, model beside the original Current Policies value."""
    part = frame.loc[frame.category.eq(category) & frame.target.notna()].copy()
    if category == "capacity":
        # Wind is the sum of the two wind rows. CSP gigawatts are inside solar-all.
        part = part.loc[~part.name.isin(["wind", "csp"])]
    if part.empty:
        return
    names = _ordered_names(part, category)
    years = [int(year) for year in sorted(part.year.unique())]
    fig_h = max(4.2, 0.36 * len(names) + 1.6)
    fig, axes = plt.subplots(
        1,
        len(years),
        figsize=(4.4 * len(years), fig_h),
        sharey=True,
        constrained_layout=True,
    )
    if len(years) == 1:
        axes = [axes]
    positions = list(range(len(names)))
    for ax, year in zip(axes, years, strict=True):
        block = part.loc[part.year.eq(year)].drop_duplicates("name").set_index("name").reindex(names)
        ax.barh(
            [item - 0.18 for item in positions],
            block.model.fillna(0.0),
            height=0.36,
            color=MODEL_COLOR,
            label="Model",
        )
        ax.barh(
            [item + 0.18 for item in positions],
            block.target.fillna(0.0),
            height=0.36,
            color=CPS_COLOR,
            label="IEA CPS",
        )
        ax.set_title(str(year))
        ax.set_xlabel(str(block.unit.dropna().iloc[0]) if block.unit.notna().any() else "")
        _style_axis(ax)
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{category} against WEO 2019, with 2018 the historical column")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_primary(frame: pd.DataFrame, path: Path) -> None:
    """Primary oil, gas and coal, each in its own unit."""
    part = frame.loc[frame.category.eq("primary") & frame.target.notna()].copy()
    if part.empty:
        return
    fuels = _ordered_names(part, "primary") or list(dict.fromkeys(part.name))
    fig, axes = plt.subplots(
        1, len(fuels), figsize=(4.0 * len(fuels), 3.8), constrained_layout=True
    )
    if len(fuels) == 1:
        axes = [axes]
    for ax, name in zip(axes, fuels, strict=True):
        block = part.loc[part.name.eq(name)].sort_values("year")
        positions = list(range(len(block)))
        ax.bar(
            [item - 0.18 for item in positions],
            block.model,
            width=0.36,
            color=MODEL_COLOR,
            label="Model",
        )
        ax.bar(
            [item + 0.18 for item in positions],
            block.target,
            width=0.36,
            color=CPS_COLOR,
            label="IEA CPS",
        )
        ax.set_xticks(positions, [str(int(year)) for year in block.year])
        ax.set_ylabel(str(block.unit.iloc[0]))
        ax.set_title(str(name))
        _style_axis(ax)
    axes[-1].legend(frameon=False)
    fig.suptitle("Primary fuel against WEO 2019, with 2018 the historical column")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_costs(frame: pd.DataFrame, path: Path) -> None:
    """Model investment and operation. There is no euro total in the outlook."""
    totals = frame.loc[
        frame.category.eq("cost")
        & frame.name.isin(["new investment", "operation", "existing capital"])
    ]
    carriers = frame.loc[frame.category.eq("cost carrier")]
    if totals.empty and carriers.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    if not totals.empty:
        years = [int(year) for year in sorted(totals.year.unique())]
        positions = list(range(len(years)))
        width = 0.24
        pieces = (
            ("operation", "Operation", MODEL_COLOR, -width),
            ("new investment", "New investment", "#4c78a8", 0.0),
            ("existing capital", "Existing capital", CPS_COLOR, width),
        )
        for name, label, color, shift in pieces:
            block = totals.loc[totals.name.eq(name)].drop_duplicates("year").set_index("year").reindex(years)
            axes[0].bar(
                [item + shift for item in positions],
                block.model.fillna(0.0),
                width=width,
                color=color,
                label=label,
            )
        axes[0].set_xticks(positions, [str(year) for year in years])
        axes[0].set_ylabel("billion euro per year")
        axes[0].set_title("System cost")
        axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
        _style_axis(axes[0])
    if not carriers.empty:
        ranked = (
            carriers.groupby("name").model.sum().sort_values(ascending=False)
        )
        keep = list(ranked.head(8).index)
        years = [int(year) for year in sorted(carriers.year.unique())]
        positions = list(range(len(years)))
        bottoms = [0.0] * len(years)
        colors = ["#1f4e79", "#4c78a8", "#72b7b2", "#54a24b", "#eeca3b", "#f58518", "#e45756", "#b279a2"]
        for color, name in zip(colors, keep, strict=False):
            block = carriers.loc[carriers.name.eq(name)].drop_duplicates("year").set_index("year").reindex(years)
            values = block.model.fillna(0.0).tolist()
            axes[1].bar(positions, values, bottom=bottoms, color=color, label=name)
            bottoms = [base + value for base, value in zip(bottoms, values, strict=True)]
        axes[1].set_xticks(positions, [str(year) for year in years])
        axes[1].set_ylabel("billion euro per year")
        axes[1].set_title("Operation plus new investment")
        axes[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        _style_axis(axes[1])
    fig.suptitle("Model cost. Current Policies does not publish this total in euro.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(networks: list[Path], config: dict, output: Path) -> pd.DataFrame:
    """Compare every solved year and write the table and graphs."""
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in networks:
        year = int(path.stem.split("_")[-1])
        network = pypsa.Network(path)
        frames.append(compare_network(network, config, year))
    report = pd.concat(frames, ignore_index=True)
    report.to_csv(output / "eu_comparison.csv", index=False)
    for category, filename in (
        ("electricity", "electricity.png"),
        ("capacity", "capacity.png"),
        ("co2", "co2.png"),
        ("final energy", "final_energy.png"),
    ):
        plot_category(report, category, output / filename)
    plot_primary(report, output / "primary.png")
    plot_costs(report, output / "costs.png")
    lines = []
    for row in report.itertuples(index=False):
        target = "n/a" if pd.isna(row.target) else f"{row.target:.3f}"
        gap = "n/a" if pd.isna(row.gap) else f"{row.gap:.3f}"
        lines.append(
            f"{int(row.year)} {row.category} {row.name}: "
            f"model {row.model:.3f} {row.unit}, outlook {target} {row.unit}, "
            f"gap {gap}, {row.status}. {row.note}".rstrip()
        )
    counts = report.status.value_counts()
    lines.append(
        "Totals: "
        + ", ".join(f"{counts.get(name, 0)} {name}" for name in ("pass", "fail", "check"))
        + "."
    )
    text = "\n".join(lines) + "\n"
    (output / "eu_comparison.txt").write_text(text)
    logger.info("Wrote %s", output / "eu_comparison.csv")
    return report


def run_snakemake(snakemake) -> None:
    """Entry point used by the workflow after every horizon has been solved."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    networks = snakemake.input.networks
    if isinstance(networks, str):
        networks = [networks]
    write_report([Path(path) for path in networks], load_config(), Path(snakemake.output.csv).parent)


def main() -> int:
    """Compare the solved horizons with the Current Policies EU totals."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("networks", nargs="*", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/iea-cps/iea_comparison"))
    args = parser.parse_args()
    networks = args.networks or [
        Path(f"results/iea-cps/networks/solved_{year}.nc")
        for year in (2018, 2030, 2035, 2040)
    ]
    missing = [path for path in networks if not path.exists()]
    if missing:
        logger.error("Solved networks are not ready: %s", ", ".join(str(path) for path in missing))
        return 2
    report = write_report(networks, load_config(args.config), args.output)
    failed = int(report.status.eq("fail").sum())
    logger.info("%s", (args.output / "eu_comparison.txt").read_text().splitlines()[-1])
    return 1 if failed else 0


if __name__ == "__main__":
    if "snakemake" in globals():
        run_snakemake(snakemake)
    else:
        raise SystemExit(main())
